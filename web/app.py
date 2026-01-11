"""Flask application for Ultimate RAG frontend."""

import os
import json
import random
from pathlib import Path
from typing import Optional, Any
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from loguru import logger
import uuid

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultimate_rag import UltimateRAG, Config
from ultimate_rag.utils.io import load_json, save_json

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ultimate-rag-secret-key-2024')
app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent / 'data' / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

# Global RAG instance
rag_instance: Optional[UltimateRAG] = None
knowledge_base: Optional[dict] = None


def get_rag() -> Optional[UltimateRAG]:
    """Get or create RAG instance."""
    global rag_instance
    if rag_instance is None:
        config = Config()
        config.generation.llm_provider = "none"
        rag_instance = UltimateRAG(config)
    return rag_instance


def load_knowledge_base_data(kb_path: Path) -> dict:
    """Load knowledge base data."""
    global knowledge_base
    knowledge_base = load_json(kb_path)
    return knowledge_base


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a PDF document."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))

        rag = get_rag()
        result = rag.process_document(filepath)

        if result.get('success'):
            session['document_id'] = result['document_id']
            session['knowledge_path'] = str(
                rag.config.processed_dir / 'knowledge' / f"{result['document_id']}_knowledge.json"
            )
            load_knowledge_base_data(Path(session['knowledge_path']))

            return jsonify({
                'success': True,
                'document_id': result['document_id'],
                'stats': {
                    'chunks': result.get('chunks', 0),
                    'formulas': result.get('formulas', 0),
                    'processing_time': f"{result.get('processing_time_s', 0):.1f}s"
                }
            })
        else:
            return jsonify({'error': result.get('error', 'Processing failed')}), 500

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_demo', methods=['POST'])
def load_demo():
    """Load demo knowledge base."""
    global knowledge_base

    # Create sample knowledge base for demo
    knowledge_base = create_demo_knowledge_base()
    session['document_id'] = 'demo_document'

    return jsonify({
        'success': True,
        'document_id': 'demo_document',
        'stats': {
            'chunks': len(knowledge_base.get('chunks', {})),
            'formulas': len(knowledge_base.get('formulas', {})),
            'tables': len(knowledge_base.get('tables', []))
        }
    })


@app.route('/api/content', methods=['POST'])
def get_content():
    """Get content based on selected types."""
    global knowledge_base

    if not knowledge_base:
        return jsonify({'error': 'No document loaded'}), 400

    data = request.json
    content_types = data.get('types', ['text', 'diagrams', 'tables'])
    page = data.get('page', 1)
    per_page = data.get('per_page', 10)

    results = {
        'text': [],
        'diagrams': [],
        'tables': [],
        'formulas': []
    }

    if 'text' in content_types:
        chunks = list(knowledge_base.get('chunks', {}).values())
        start = (page - 1) * per_page
        end = start + per_page

        for chunk in chunks[start:end]:
            results['text'].append({
                'id': chunk.get('id'),
                'content': chunk.get('content', '')[:500],
                'type': chunk.get('chunk_type', 'paragraph'),
                'page': chunk.get('metadata', {}).get('page_numbers', [None])[0],
                'heading': chunk.get('metadata', {}).get('heading', ''),
                'domain': chunk.get('metadata', {}).get('domain', 'general')
            })

        # Include formulas with text
        formulas = list(knowledge_base.get('formulas', {}).values())
        for formula in formulas[:20]:
            results['formulas'].append({
                'id': formula.get('id'),
                'latex': formula.get('latex', ''),
                'plain_text': formula.get('plain_text', ''),
                'domain': formula.get('domain', 'general'),
                'page': formula.get('source_page')
            })

    if 'diagrams' in content_types:
        diagrams = knowledge_base.get('diagrams', [])
        for diagram in diagrams[:per_page]:
            results['diagrams'].append({
                'id': diagram.get('id'),
                'type': diagram.get('diagram_type', 'figure'),
                'caption': diagram.get('caption', ''),
                'page': diagram.get('metadata', {}).get('page')
            })

    if 'tables' in content_types:
        tables = knowledge_base.get('tables', [])
        for table in tables[:per_page]:
            results['tables'].append({
                'id': table.get('id'),
                'headers': table.get('headers', []),
                'rows': table.get('rows', [])[:5],
                'page': table.get('page_number'),
                'total_rows': len(table.get('rows', []))
            })

    return jsonify({
        'success': True,
        'results': results,
        'total': {
            'text': len(knowledge_base.get('chunks', {})),
            'diagrams': len(knowledge_base.get('diagrams', [])),
            'tables': len(knowledge_base.get('tables', [])),
            'formulas': len(knowledge_base.get('formulas', {}))
        }
    })


@app.route('/api/flashcards', methods=['POST'])
def generate_flashcards():
    """Generate flashcards from content."""
    global knowledge_base

    if not knowledge_base:
        return jsonify({'error': 'No document loaded'}), 400

    data = request.json
    count = min(data.get('count', 10), 50)
    domain = data.get('domain', 'all')

    flashcards = []

    # Generate flashcards from formulas
    formulas = list(knowledge_base.get('formulas', {}).values())
    if domain != 'all':
        formulas = [f for f in formulas if f.get('domain') == domain]

    for formula in formulas[:count // 2]:
        flashcards.append({
            'id': f"fc_{formula.get('id')}",
            'type': 'formula',
            'front': f"What is the formula for: {formula.get('plain_text', formula.get('description', 'this concept'))}?",
            'back': f"$${formula.get('latex', '')}$$",
            'domain': formula.get('domain', 'general'),
            'difficulty': random.choice(['easy', 'medium', 'hard'])
        })

    # Generate flashcards from chunks (definitions, theorems)
    chunks = list(knowledge_base.get('chunks', {}).values())
    definition_chunks = [c for c in chunks if c.get('chunk_type') in ['definition', 'theorem', 'concept']]

    if domain != 'all':
        definition_chunks = [c for c in definition_chunks if c.get('metadata', {}).get('domain') == domain]

    for chunk in definition_chunks[:count // 2]:
        content = chunk.get('content', '')
        heading = chunk.get('metadata', {}).get('heading', '')

        if heading:
            flashcards.append({
                'id': f"fc_{chunk.get('id')}",
                'type': 'concept',
                'front': f"Define or explain: {heading}",
                'back': content[:300] + ('...' if len(content) > 300 else ''),
                'domain': chunk.get('metadata', {}).get('domain', 'general'),
                'difficulty': random.choice(['easy', 'medium', 'hard'])
            })

    # Shuffle flashcards
    random.shuffle(flashcards)

    return jsonify({
        'success': True,
        'flashcards': flashcards[:count],
        'total_available': len(formulas) + len(definition_chunks)
    })


@app.route('/api/mcq', methods=['POST'])
def generate_mcq():
    """Generate MCQ questions from content."""
    global knowledge_base

    if not knowledge_base:
        return jsonify({'error': 'No document loaded'}), 400

    data = request.json
    count = min(data.get('count', 10), 30)
    domain = data.get('domain', 'all')
    difficulty = data.get('difficulty', 'mixed')

    questions = []

    # Generate MCQs from formulas
    formulas = list(knowledge_base.get('formulas', {}).values())
    if domain != 'all':
        formulas = [f for f in formulas if f.get('domain') == domain]

    for i, formula in enumerate(formulas[:count // 2]):
        # Create "identify the formula" questions
        other_formulas = [f for f in formulas if f.get('id') != formula.get('id')]
        wrong_answers = random.sample(
            [f.get('latex', '') for f in other_formulas][:10] or ['E = hf', 'F = ma', 'PV = nRT'],
            min(3, len(other_formulas) or 3)
        )

        options = wrong_answers + [formula.get('latex', '')]
        random.shuffle(options)
        correct_index = options.index(formula.get('latex', ''))

        questions.append({
            'id': f"mcq_{formula.get('id')}",
            'type': 'formula_identification',
            'question': f"Which formula represents {formula.get('plain_text', formula.get('description', 'this concept'))}?",
            'options': [f"$${opt}$$" for opt in options],
            'correct': correct_index,
            'explanation': f"The correct formula is $${formula.get('latex', '')}$$",
            'domain': formula.get('domain', 'general'),
            'difficulty': difficulty if difficulty != 'mixed' else random.choice(['easy', 'medium', 'hard'])
        })

    # Generate MCQs from concepts
    chunks = list(knowledge_base.get('chunks', {}).values())
    concept_chunks = [c for c in chunks if c.get('chunk_type') in ['definition', 'concept', 'theorem']]

    if domain != 'all':
        concept_chunks = [c for c in concept_chunks if c.get('metadata', {}).get('domain') == domain]

    for chunk in concept_chunks[:count // 2]:
        heading = chunk.get('metadata', {}).get('heading', '')
        content = chunk.get('content', '')

        if heading and content:
            # Create true/false or concept questions
            questions.append({
                'id': f"mcq_{chunk.get('id')}",
                'type': 'concept',
                'question': f"Which of the following best describes '{heading}'?",
                'options': [
                    content[:150] + ('...' if len(content) > 150 else ''),
                    "This concept is not related to the subject matter.",
                    "This is a mathematical constant with no physical meaning.",
                    "This describes a phenomenon that has been disproven."
                ],
                'correct': 0,
                'explanation': content[:200],
                'domain': chunk.get('metadata', {}).get('domain', 'general'),
                'difficulty': difficulty if difficulty != 'mixed' else random.choice(['easy', 'medium', 'hard'])
            })

    random.shuffle(questions)

    return jsonify({
        'success': True,
        'questions': questions[:count],
        'total_available': len(formulas) + len(concept_chunks)
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot queries."""
    global knowledge_base

    data = request.json
    message = data.get('message', '').strip()
    conversation_id = data.get('conversation_id', str(uuid.uuid4()))

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    # Initialize conversation history in session
    if 'conversations' not in session:
        session['conversations'] = {}

    if conversation_id not in session['conversations']:
        session['conversations'][conversation_id] = []

    history = session['conversations'][conversation_id]
    history.append({'role': 'user', 'content': message})

    # Generate response
    response_data = generate_chat_response(message, knowledge_base, history)

    history.append({'role': 'assistant', 'content': response_data['response']})
    session['conversations'][conversation_id] = history[-20:]  # Keep last 20 messages
    session.modified = True

    return jsonify({
        'success': True,
        'response': response_data['response'],
        'sources': response_data.get('sources', []),
        'suggestions': response_data.get('suggestions', []),
        'conversation_id': conversation_id
    })


@app.route('/api/search', methods=['POST'])
def search():
    """Search the knowledge base."""
    global knowledge_base

    if not knowledge_base:
        return jsonify({'error': 'No document loaded'}), 400

    data = request.json
    query = data.get('query', '').strip().lower()

    if not query:
        return jsonify({'error': 'Empty query'}), 400

    results = {
        'chunks': [],
        'formulas': [],
        'tables': []
    }

    # Search chunks
    for chunk_id, chunk in knowledge_base.get('chunks', {}).items():
        content = chunk.get('content', '').lower()
        heading = chunk.get('metadata', {}).get('heading', '').lower()

        if query in content or query in heading:
            results['chunks'].append({
                'id': chunk_id,
                'content': chunk.get('content', '')[:300],
                'heading': chunk.get('metadata', {}).get('heading', ''),
                'type': chunk.get('chunk_type'),
                'relevance': content.count(query) + heading.count(query) * 2
            })

    # Search formulas
    for formula_id, formula in knowledge_base.get('formulas', {}).items():
        latex = formula.get('latex', '').lower()
        plain = formula.get('plain_text', '').lower()
        desc = formula.get('description', '').lower()

        if query in latex or query in plain or query in desc:
            results['formulas'].append({
                'id': formula_id,
                'latex': formula.get('latex', ''),
                'plain_text': formula.get('plain_text', ''),
                'domain': formula.get('domain', 'general')
            })

    # Sort by relevance
    results['chunks'].sort(key=lambda x: x.get('relevance', 0), reverse=True)

    return jsonify({
        'success': True,
        'results': results,
        'total': sum(len(v) for v in results.values())
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get document statistics."""
    global knowledge_base

    if not knowledge_base:
        return jsonify({'loaded': False})

    stats = knowledge_base.get('statistics', {})

    return jsonify({
        'loaded': True,
        'document_id': knowledge_base.get('document_id', 'unknown'),
        'chunks': stats.get('total_chunks', len(knowledge_base.get('chunks', {}))),
        'formulas': stats.get('total_formulas', len(knowledge_base.get('formulas', {}))),
        'tables': len(knowledge_base.get('tables', [])),
        'diagrams': len(knowledge_base.get('diagrams', [])),
        'domains': stats.get('domains', {}),
        'chunk_types': stats.get('chunk_types', {})
    })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_chat_response(message: str, kb: Optional[dict], history: list) -> dict:
    """Generate chatbot response based on knowledge base."""
    message_lower = message.lower()

    # Default response structure
    response_data = {
        'response': '',
        'sources': [],
        'suggestions': []
    }

    if not kb:
        response_data['response'] = (
            "I don't have any document loaded yet. Please upload a PDF document or load the demo "
            "to start exploring the content. Once loaded, I can help you with:\n\n"
            "- Explaining concepts and formulas\n"
            "- Finding specific information\n"
            "- Creating flashcards and quizzes\n"
            "- Answering questions about the material"
        )
        response_data['suggestions'] = ['Load demo document', 'Upload a PDF', 'What can you help with?']
        return response_data

    # Search for relevant content
    relevant_chunks = []
    relevant_formulas = []

    # Simple keyword matching for demo
    keywords = message_lower.split()

    for chunk_id, chunk in kb.get('chunks', {}).items():
        content = chunk.get('content', '').lower()
        score = sum(1 for kw in keywords if kw in content and len(kw) > 3)
        if score > 0:
            relevant_chunks.append((chunk, score))

    for formula_id, formula in kb.get('formulas', {}).items():
        latex = formula.get('latex', '').lower()
        plain = formula.get('plain_text', '').lower()
        score = sum(1 for kw in keywords if kw in latex or kw in plain)
        if score > 0:
            relevant_formulas.append((formula, score))

    # Sort by relevance
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    relevant_formulas.sort(key=lambda x: x[1], reverse=True)

    # Build response
    if relevant_chunks or relevant_formulas:
        response_parts = []

        if relevant_chunks:
            top_chunk = relevant_chunks[0][0]
            response_parts.append(f"Based on the document:\n\n{top_chunk.get('content', '')[:400]}")

            heading = top_chunk.get('metadata', {}).get('heading', '')
            page = top_chunk.get('metadata', {}).get('page_numbers', [None])[0]
            if heading or page:
                response_data['sources'].append({
                    'type': 'text',
                    'heading': heading,
                    'page': page
                })

        if relevant_formulas:
            top_formula = relevant_formulas[0][0]
            formula_text = f"\n\nRelevant formula:\n$${top_formula.get('latex', '')}$$"
            if top_formula.get('plain_text'):
                formula_text += f"\n({top_formula.get('plain_text')})"
            response_parts.append(formula_text)

            response_data['sources'].append({
                'type': 'formula',
                'latex': top_formula.get('latex', ''),
                'page': top_formula.get('source_page')
            })

        response_data['response'] = '\n'.join(response_parts)

        # Generate follow-up suggestions
        if relevant_chunks:
            chunk = relevant_chunks[0][0]
            domain = chunk.get('metadata', {}).get('domain', 'general')
            response_data['suggestions'] = [
                f"Tell me more about {domain}",
                "Show related formulas",
                "Create flashcards on this topic"
            ]
    else:
        response_data['response'] = (
            "I couldn't find specific information about that in the document. "
            "Could you try rephrasing your question or asking about:\n\n"
            f"- One of the {len(kb.get('formulas', {}))} formulas in the document\n"
            f"- Any of the {len(kb.get('chunks', {}))} content sections\n"
            "- Specific concepts or definitions"
        )
        response_data['suggestions'] = [
            "Show all formulas",
            "List main topics",
            "Generate practice questions"
        ]

    return response_data


def create_demo_knowledge_base() -> dict:
    """Create a demo knowledge base with sample content."""
    return {
        'document_id': 'demo_document',
        'chunks': {
            'chunk_001': {
                'id': 'chunk_001',
                'content': "Newton's Second Law of Motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. This fundamental principle forms the basis of classical mechanics and is expressed mathematically as F = ma, where F represents force, m represents mass, and a represents acceleration.",
                'chunk_type': 'definition',
                'metadata': {
                    'heading': "Newton's Second Law",
                    'page_numbers': [15],
                    'domain': 'physics',
                    'concepts': ['force', 'mass', 'acceleration', 'Newton']
                }
            },
            'chunk_002': {
                'id': 'chunk_002',
                'content': "The Law of Conservation of Energy states that energy cannot be created or destroyed, only transformed from one form to another. In an isolated system, the total energy remains constant. This principle is fundamental to all of physics and has applications in mechanics, thermodynamics, and quantum physics.",
                'chunk_type': 'theorem',
                'metadata': {
                    'heading': 'Conservation of Energy',
                    'page_numbers': [42],
                    'domain': 'physics',
                    'concepts': ['energy', 'conservation', 'thermodynamics']
                }
            },
            'chunk_003': {
                'id': 'chunk_003',
                'content': "The quadratic formula provides the solutions to any quadratic equation of the form ax² + bx + c = 0. The solutions are given by x = (-b ± √(b² - 4ac)) / 2a. The discriminant, b² - 4ac, determines the nature of the roots: positive means two real roots, zero means one repeated root, negative means complex roots.",
                'chunk_type': 'concept',
                'metadata': {
                    'heading': 'Quadratic Formula',
                    'page_numbers': [78],
                    'domain': 'mathematics',
                    'concepts': ['quadratic', 'roots', 'discriminant', 'algebra']
                }
            },
            'chunk_004': {
                'id': 'chunk_004',
                'content': "Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent. In machine learning, it is used to find the optimal parameters that minimize the loss function. The update rule is: θ = θ - α∇J(θ), where α is the learning rate and ∇J(θ) is the gradient of the cost function.",
                'chunk_type': 'concept',
                'metadata': {
                    'heading': 'Gradient Descent',
                    'page_numbers': [156],
                    'domain': 'data_science',
                    'concepts': ['optimization', 'gradient', 'machine learning', 'loss function']
                }
            },
            'chunk_005': {
                'id': 'chunk_005',
                'content': "Le Chatelier's Principle states that if a dynamic equilibrium is disturbed by changing the conditions, the position of equilibrium shifts to counteract the change. This principle helps predict how changes in concentration, temperature, or pressure will affect a chemical reaction at equilibrium.",
                'chunk_type': 'theorem',
                'metadata': {
                    'heading': "Le Chatelier's Principle",
                    'page_numbers': [203],
                    'domain': 'chemistry',
                    'concepts': ['equilibrium', 'reaction', 'Le Chatelier']
                }
            },
            'chunk_006': {
                'id': 'chunk_006',
                'content': "Maxwell's Equations are a set of four fundamental equations that describe how electric and magnetic fields are generated and altered by each other and by charges and currents. They form the foundation of classical electromagnetism and predict the existence of electromagnetic waves traveling at the speed of light.",
                'chunk_type': 'definition',
                'metadata': {
                    'heading': "Maxwell's Equations",
                    'page_numbers': [89],
                    'domain': 'physics',
                    'concepts': ['Maxwell', 'electromagnetic', 'electric field', 'magnetic field']
                }
            },
            'chunk_007': {
                'id': 'chunk_007',
                'content': "The Central Limit Theorem states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution. This theorem is fundamental to statistical inference and hypothesis testing.",
                'chunk_type': 'theorem',
                'metadata': {
                    'heading': 'Central Limit Theorem',
                    'page_numbers': [134],
                    'domain': 'data_science',
                    'concepts': ['statistics', 'normal distribution', 'sampling', 'CLT']
                }
            },
            'chunk_008': {
                'id': 'chunk_008',
                'content': "The Ideal Gas Law combines several gas laws into one equation: PV = nRT, where P is pressure, V is volume, n is the number of moles, R is the gas constant, and T is temperature. This equation describes the behavior of ideal gases and is widely used in chemistry and physics.",
                'chunk_type': 'concept',
                'metadata': {
                    'heading': 'Ideal Gas Law',
                    'page_numbers': [167],
                    'domain': 'chemistry',
                    'concepts': ['gas', 'pressure', 'volume', 'temperature', 'ideal gas']
                }
            }
        },
        'formulas': {
            'formula_001': {
                'id': 'formula_001',
                'latex': 'F = ma',
                'plain_text': "Newton's Second Law: Force equals mass times acceleration",
                'domain': 'physics',
                'source_page': 15,
                'formula_type': 'law'
            },
            'formula_002': {
                'id': 'formula_002',
                'latex': 'E = mc^2',
                'plain_text': "Einstein's mass-energy equivalence",
                'domain': 'physics',
                'source_page': 45,
                'formula_type': 'law'
            },
            'formula_003': {
                'id': 'formula_003',
                'latex': 'x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}',
                'plain_text': 'Quadratic formula for solving ax² + bx + c = 0',
                'domain': 'mathematics',
                'source_page': 78,
                'formula_type': 'formula'
            },
            'formula_004': {
                'id': 'formula_004',
                'latex': '\\theta = \\theta - \\alpha \\nabla J(\\theta)',
                'plain_text': 'Gradient descent update rule',
                'domain': 'data_science',
                'source_page': 156,
                'formula_type': 'algorithm'
            },
            'formula_005': {
                'id': 'formula_005',
                'latex': 'PV = nRT',
                'plain_text': 'Ideal Gas Law',
                'domain': 'chemistry',
                'source_page': 167,
                'formula_type': 'law'
            },
            'formula_006': {
                'id': 'formula_006',
                'latex': 'KE = \\frac{1}{2}mv^2',
                'plain_text': 'Kinetic Energy formula',
                'domain': 'physics',
                'source_page': 23,
                'formula_type': 'formula'
            },
            'formula_007': {
                'id': 'formula_007',
                'latex': '\\int_a^b f(x)\\,dx = F(b) - F(a)',
                'plain_text': 'Fundamental Theorem of Calculus',
                'domain': 'mathematics',
                'source_page': 92,
                'formula_type': 'theorem'
            },
            'formula_008': {
                'id': 'formula_008',
                'latex': '\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}',
                'plain_text': "Gauss's Law for electricity",
                'domain': 'physics',
                'source_page': 89,
                'formula_type': 'law'
            },
            'formula_009': {
                'id': 'formula_009',
                'latex': 'H = -\\sum_{i} p_i \\log_2 p_i',
                'plain_text': 'Shannon Entropy formula',
                'domain': 'data_science',
                'source_page': 178,
                'formula_type': 'formula'
            },
            'formula_010': {
                'id': 'formula_010',
                'latex': '\\Delta G = \\Delta H - T\\Delta S',
                'plain_text': 'Gibbs Free Energy equation',
                'domain': 'chemistry',
                'source_page': 198,
                'formula_type': 'formula'
            }
        },
        'tables': [
            {
                'id': 'table_001',
                'headers': ['Quantity', 'Symbol', 'SI Unit'],
                'rows': [
                    ['Force', 'F', 'Newton (N)'],
                    ['Mass', 'm', 'Kilogram (kg)'],
                    ['Acceleration', 'a', 'm/s²'],
                    ['Energy', 'E', 'Joule (J)'],
                    ['Power', 'P', 'Watt (W)']
                ],
                'page_number': 12
            },
            {
                'id': 'table_002',
                'headers': ['Element', 'Symbol', 'Atomic Number', 'Atomic Mass'],
                'rows': [
                    ['Hydrogen', 'H', '1', '1.008'],
                    ['Carbon', 'C', '6', '12.011'],
                    ['Nitrogen', 'N', '7', '14.007'],
                    ['Oxygen', 'O', '8', '15.999']
                ],
                'page_number': 165
            }
        ],
        'diagrams': [
            {
                'id': 'diagram_001',
                'diagram_type': 'vector_diagram',
                'caption': 'Force vectors acting on an object',
                'metadata': {'page': 16}
            },
            {
                'id': 'diagram_002',
                'diagram_type': 'graph',
                'caption': 'Gradient descent optimization path',
                'metadata': {'page': 157}
            }
        ],
        'statistics': {
            'total_chunks': 8,
            'total_formulas': 10,
            'domains': {
                'physics': 4,
                'mathematics': 2,
                'chemistry': 2,
                'data_science': 2
            },
            'chunk_types': {
                'definition': 2,
                'theorem': 3,
                'concept': 3
            }
        }
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    logger.info("Starting Ultimate RAG Web Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
