"""Visualization utility functions."""

from pathlib import Path
from typing import Optional, Any
from loguru import logger


def render_latex(latex: str, output_path: Optional[Path] = None) -> Optional[bytes]:
    """Render LaTeX formula to image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO

        fig, ax = plt.subplots(figsize=(6, 1))
        ax.axis('off')
        ax.text(
            0.5, 0.5, f"${latex}$",
            fontsize=16, ha='center', va='center',
            transform=ax.transAxes
        )

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   pad_inches=0.1, dpi=150, transparent=True)
        plt.close(fig)

        image_bytes = buf.getvalue()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(image_bytes)

        return image_bytes

    except Exception as e:
        logger.error(f"LaTeX rendering failed: {e}")
        return None


def create_chunk_graph(
    chunks: list[Any],
    output_path: Optional[Path] = None
) -> Optional[str]:
    """Create visualization of chunk relationships."""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        for chunk in chunks:
            label = chunk.content[:30] + "..." if len(chunk.content) > 30 else chunk.content
            G.add_node(chunk.id, label=label, type=chunk.chunk_type.value)

            if chunk.parent_id:
                G.add_edge(chunk.parent_id, chunk.id)

            for related_id in chunk.related_ids:
                G.add_edge(chunk.id, related_id, style='dashed')

        fig, ax = plt.subplots(figsize=(12, 8))

        pos = nx.spring_layout(G, k=2, iterations=50)

        type_colors = {
            'chapter': '#ff6b6b',
            'section': '#4ecdc4',
            'concept': '#45b7d1',
            'paragraph': '#96ceb4',
            'formula_block': '#ffeaa7',
            'default': '#dfe6e9'
        }

        colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'default')
            colors.append(type_colors.get(node_type, type_colors['default']))

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, ax=ax)

        labels = {n: G.nodes[n].get('label', n)[:15] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title("Chunk Relationship Graph")
        ax.axis('off')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return str(output_path)

        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()

    except ImportError:
        logger.warning("networkx/matplotlib not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Graph visualization failed: {e}")
        return None


def visualize_results(
    results: list[Any],
    query: str,
    output_path: Optional[Path] = None
) -> Optional[str]:
    """Visualize search results with scores."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        labels = []
        scores = []
        colors = []

        source_colors = {
            'text': '#3498db',
            'formula': '#e74c3c',
            'diagram': '#2ecc71',
            'keyword': '#f39c12'
        }

        for result in results[:10]:
            label = result.content[:40] + "..." if len(result.content) > 40 else result.content
            labels.append(label)
            scores.append(result.score)
            colors.append(source_colors.get(result.source_type, '#95a5a6'))

        y_pos = range(len(labels))

        ax.barh(y_pos, scores, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Relevance Score')
        ax.set_title(f'Search Results for: "{query[:50]}..."' if len(query) > 50 else f'Search Results for: "{query}"')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=t.title())
            for t, c in source_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return str(output_path)

        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()

    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Results visualization failed: {e}")
        return None


def create_formula_dependency_graph(
    formulas: list[Any],
    output_path: Optional[Path] = None
) -> Optional[str]:
    """Create visualization of formula dependencies."""
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        for formula in formulas:
            label = formula.latex[:20] + "..." if len(formula.latex) > 20 else formula.latex
            G.add_node(formula.id, label=label, domain=formula.domain or 'general')

            for dep_id in formula.dependencies:
                G.add_edge(dep_id, formula.id)

        fig, ax = plt.subplots(figsize=(12, 8))

        pos = nx.kamada_kawai_layout(G)

        domain_colors = {
            'physics': '#e74c3c',
            'mathematics': '#3498db',
            'chemistry': '#2ecc71',
            'general': '#95a5a6'
        }

        colors = []
        for node in G.nodes():
            domain = G.nodes[node].get('domain', 'general')
            colors.append(domain_colors.get(domain, domain_colors['general']))

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                               arrowsize=15, ax=ax)

        labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title("Formula Dependency Graph")
        ax.axis('off')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return str(output_path)

        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()

    except ImportError:
        logger.warning("Dependencies not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Formula graph visualization failed: {e}")
        return None
