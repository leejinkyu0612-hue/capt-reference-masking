"""Figures for the manuscript (grayscale-safe, IEEE single-column width 3.45 in).
Usage: python figures.py <out_dir> [analysis_json]"""
import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1]
AJ = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
W = 3.45


def box(ax, x, y, w, h, text, fc='white', lw=0.8, fs=6.4, bold=False, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='black', lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.15)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.7, color='black'))


def fig1():
    fig, ax = plt.subplots(figsize=(W, 2.05)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    box(ax, 0.05, 3.9, 2.2, 1.5, 'Learner audio\n(read-aloud\nutterance)', fc='#f2f2f2')
    box(ax, 0.05, 1.0, 2.2, 1.5, 'Target sentence\n"ALICE GIVE\nUP BOXING"', fc='#f2f2f2')
    box(ax, 3.0, 2.3, 2.5, 2.4, 'Transcriber S\nWhisper (CPU)\nTranscription API\nAudio LLM', fc='white', bold=False)
    ax.text(4.25, 5.35, 'Setting c = (R, V)', ha='center', fontsize=7.5, style='italic')
    arrow(ax, 2.25, 4.65, 3.0, 4.0)
    arrow(ax, 2.25, 1.75, 3.0, 3.0)
    ax.text(2.45, 2.55, 'R', fontsize=7.5, fontweight='bold')
    box(ax, 6.2, 3.9, 3.7, 1.5, 'Transcript h aligned\nto target words\n(accept / reject)', fc='white')
    arrow(ax, 5.5, 3.9, 6.2, 4.55)
    box(ax, 6.2, 1.0, 3.7, 1.9, 'Expert labels (5 raters)\nmispronounced → FAR\ncorrect → FRR', fc='#e6e6e6')
    arrow(ax, 8.05, 3.9, 8.05, 2.9)
    ax.text(4.25, 0.55, 'R: target sentence given as context;  V: verbatim instruction (audio LLM)', ha='center', fontsize=6.8)
    fig.savefig(os.path.join(OUT, 'Fig1_pipeline.png'), dpi=600, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'Fig1_pipeline.pdf'), bbox_inches='tight')
    plt.close(fig)


LABEL = {'whisper-tiny': 'Whisper tiny', 'whisper-base': 'Whisper base', 'whisper-small': 'Whisper small',
         'gpt-4o-mini-transcribe': 'gpt-4o-mini-transcribe', 'gemini-2.5-flash': 'gemini-2.5-flash', 'gpt-audio-mini': 'gpt-audio-mini'}


def fig2(a):
    rows = [c for c in a['confirmatory'] if c['hyp'] in ('H1', 'H2a')]
    names = [LABEL[c['A'].split('_')[0]] for c in rows]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, key, title in ((axes[0], 'FAR', 'False acceptance (FAR, %)'), (axes[1], 'FRR', 'False rejection (FRR, %)')):
        for i, c in enumerate(rows):
            y = len(rows) - 1 - i
            a0, b0 = 100 * c[key + '_A'], 100 * c[key + '_B']
            ax.plot([a0, b0], [y, y], color='#9a9a9a', lw=2.2, solid_capstyle='round', zorder=1)
            ax.scatter([a0], [y], s=22, facecolor='white', edgecolor='black', lw=0.8, zorder=3, label='R0 (no context)' if i == 0 else None)
            ax.scatter([b0], [y], s=22, facecolor='black', edgecolor='black', lw=0.8, zorder=3, label='R1 (target sentence)' if i == 0 else None)
            d = b0 - a0
            ax.text(max(a0, b0) + 3, y, f'{d:+.1f}', va='center', fontsize=6.5)
        ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_title(title, fontsize=7.5, pad=3)
        ax.grid(axis='x', color='#dddddd', lw=0.5); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(names[::-1], fontsize=7)
    axes[0].legend(loc='lower center', bbox_to_anchor=(1.0, -0.42), ncol=2, frameon=False, fontsize=6.8, handletextpad=0.3)
    fig.savefig(os.path.join(OUT, 'Fig2_reference_context.png'), dpi=600, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'Fig2_reference_context.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig1()
    if AJ:
        fig2(json.load(open(AJ, encoding='utf-8')))
    print('figures ->', OUT)

