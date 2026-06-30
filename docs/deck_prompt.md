# Prompt: build the slide deck

Paste everything below the line into Claude (claude.ai), ideally with the seven PNGs from
`docs/figures/` attached. It will produce a self-contained HTML slide deck. All numbers are
real (from this project) - do not change them.

---

You are a presentation designer. Build a polished, self-contained **HTML slide deck** (single
`.html` file, embedded CSS, no external dependencies, keyboard arrow navigation, one slide per
full screen, 16:9). Dark, modern, technical aesthetic (deep navy background, one accent
colour, large readable type, generous whitespace). Each slide must fit on screen without
scrolling. Use the attached figures where indicated (reference them by filename; I will place
the images next to the HTML).

This is an academic project deck for an ESADE MSc "Recommender Systems" course (Prof. Marc
Torrens). The grade splits 50% technical implementation, 30% evaluation, 20% UX, so the deck
must foreground **method comparison, evaluation, and critical thinking** - not just "it
works". Keep prose tight (headline + 3-5 bullets per slide). Do not invent numbers; use only
the figures and tables given here.

## Project in one line
A music recommender prototype on the **Last.fm HetRec 2011** implicit-feedback dataset, built
module by module, comparing eight methods on the same data across accuracy AND beyond-accuracy
metrics.

## Slides

**1. Title.** "Music Recommender: Eight Methods, One Honest Comparison". Subtitle: Last.fm
HetRec 2011, implicit feedback. Course / author / date placeholder.

**2. The problem & the data.** Implicit feedback only (play counts, no star ratings). Last.fm
HetRec 2011: **1,892 users x 17,632 artists, 92,834 interactions, 99.72% sparse**. Key design
consequence: everything is built around confidence from play counts, and evaluation is
ranking-based (not rating prediction). Mention also available: artist tags (11,946 tags,
186k assignments) and a friendship graph.

**3. Know your data (EDA).** Use figure `long_tail_lorenz.png`. Headline: **listening is
extremely concentrated - Gini = 0.893**; the top 100 artists capture 43.7% of all plays;
median artist has just 1 listener (a huge long tail). This is the evidence behind "accuracy
is not enough": a popularity recommender can score acceptably while ignoring ~99% of the
catalogue. Optionally also reference `play_count_distribution.png` (raw plays span orders of
magnitude -> log-scaling / confidence weighting is justified).

**4. Approach: agile, module by module.** A thin end-to-end slice first (load -> popularity ->
one screen -> one metric), then deepen. Seven modules: UI, EDA/preprocessing, popularity
baseline, collaborative filtering, content-based, matrix factorisation, evaluation. Shared
`src/recsys` package; Flask prototype; every method behind one `fit/recommend` interface so
they are interchangeable. Test-driven, 86 tests, 95% coverage.

**5. Implicit-feedback design.** No "dislikes" exist. Treat plays as confidence:
`c = 1 + alpha * log(1 + plays)` (Hu, Koren, Volinsky 2008). Log-scaling stops a few superfans
dominating. Per-user leave-out split so every test user is also in training. One shared
train/test split and one evaluation harness reused by every method - the backbone of a fair
comparison.

**6. The eight methods (overview).** One line each (popularity covers three variants):
- Popularity (plays / listeners / damped) - non-personalised baseline / floor.
- Item-item CF - artists co-listened with yours (hand-rolled sparse cosine).
- User-user CF - artists from users with similar taste.
- Content-based - TF-IDF over artist tags; matches your tag profile; handles cold-start.
- Matrix factorisation (implicit ALS) - hand-implemented Hu/Koren/Volinsky; learns latent
  taste factors.
- Social - recommends what your friends listen to, using the friendship graph.
All core logic hand-implemented; numpy/scipy/sklearn for speed.

**7. How we evaluate (the 30%).** Same split, same harness, three metric families, and
every result averaged over 5 random splits (std 0.001-0.003, so rankings are stable):
- Accuracy: Precision@10, Recall@10, MAP@10, NDCG@10.
- Beyond-accuracy: catalogue coverage, intra-list diversity, novelty, popularity bias
  (mean recommended popularity + exposure Gini).
- Operational: training time, serving latency.

**8. Results: accuracy.** Use figure `eval_accuracy.png`. Headline: **personalisation more
than doubles the baseline.** Item-item CF Precision@10 = 0.175, NDCG = 0.217 vs best
popularity baseline 0.069 / 0.080 (~2.5x). Surprise worth stating: simple item-item CF beats
the more complex ALS on this small, dense dataset - sophistication is not automatically
better.

**9. Results: accuracy is not enough.** Use figure `eval_beyond_accuracy.png`. No method wins
every objective. Item-CF: best accuracy + broad coverage. ALS: most diverse of the accurate
methods. Content-based: most novel but least diverse (filter bubble). Popularity: ~0.001
coverage (the same ~25 artists to all 1,884 users).

**10. The core trade-off.** Use figure `eval_tradeoff.png` (accuracy vs diversity scatter).
Point: methods occupy different corners; choosing one means choosing what you value.

**11. Popularity bias, quantified.** Use figure `eval_scalability_bias.png` (left panel,
exposure Gini). Popularity recommenders have the highest bias (mean recommended popularity
0.186) and near-total exposure concentration; content-based is the least biased (0.034).
Caveat to show honesty: even "personalised" user-user CF leans on crowd-pleasers (0.140).

**12. Scalability.** Use figure `eval_scalability_bias.png` (right panel). Two cost profiles:
memory-based CF trains in ~0.02s but stores a big item-item similarity matrix; ALS pays ~6.3s
training but serves in ~0.2ms from compact factor matrices. No universal winner.

**13. Full comparison table.** Render this exact table (k=10, mean over 5 splits, ~1,884
users):

| Method | P@10 | NDCG | Coverage | Diversity | Novelty | PopBias | fit s |
|---|---|---|---|---|---|---|---|
| Item-item CF | 0.177 | 0.220 | 0.158 | 0.616 | 4.75 | 0.084 | 0.02 |
| ALS (MF) | 0.134 | 0.150 | 0.102 | 0.746 | 4.92 | 0.053 | 6.3 |
| User-user CF | 0.131 | 0.162 | 0.019 | 0.620 | 2.98 | 0.140 | 0.03 |
| Social (friends) | 0.123 | 0.153 | 0.143 | 0.712 | 4.84 | 0.078 | 0.03 |
| Content-based | 0.100 | 0.117 | 0.145 | 0.337 | 6.59 | 0.034 | 0.04 |
| Pop (listeners) | 0.071 | 0.082 | 0.002 | 0.632 | 2.45 | 0.186 | 0.01 |
| Pop (plays) | 0.061 | 0.066 | 0.001 | 0.698 | 2.69 | 0.165 | 0.01 |
| Pop (damped) | 0.045 | 0.046 | 0.001 | 0.753 | 3.36 | 0.120 | 0.01 |

Bold the best cell in each column.

**13b. Cold-start.** Of 2,211 artists with zero training plays, collaborative filtering,
ALS, social, and popularity can recommend NONE (not in their item space); content-based can
recommend 1,227 of them - its defining advantage. Conversely, this dataset has almost no
cold users (75% have exactly 40 training interactions), so accuracy is flat across user
history - itself an honest dataset finding.

**14. Technical challenges.** Extreme sparsity (99.7%); implicit feedback has no negatives;
keeping comparison fair (one split, one harness, interchangeable models); hand-implementing
ALS correctly (validated with a closed-form unit test + a synthetic two-cluster sanity test);
the diversity metric only meaningful in tag space; balancing accuracy against bias.

**15. Honest limitations.** State them plainly (this was explicitly requested). First, what
was *fixed* rather than excused: averaged over 5 seeds (not one); ALS tuned on a validation
split; cold-start measured; the friendship graph turned into a working method. What genuinely
*remains* (mostly inherent): no temporal split is possible because the interaction file has
no timestamps (dataset constraint, not a choice); accuracy rewards re-discovering
already-played artists, not true discovery - inherent to offline evaluation, needs an online
test to fix; ALS validated by unit tests and a synthetic check, not the `implicit` library;
diversity is tag-space only and exposure Gini is near-saturated for all methods; the dataset
is small and dense (top ~50 artists/user), which flatters the popularity baseline and leaves
almost no cold users; scalability is described at this scale only; the UI is a prototype, not
user-tested.

**16. Final remarks.** No single model wins on all axes - the right system is a **portfolio**:
ship item-item CF as the default (best accuracy + coverage, trivial training), add
content-based for cold-start and novelty, use the social recommender to widen coverage where
a friendship graph exists, and keep popularity as the cold-user fallback. Key lesson:
accuracy is necessary but not sufficient; diversity, novelty, coverage, bias, and scalability
all matter. Next steps: hybrid blending, learning-to-rank, `implicit`-library cross-check.

**17. Closing.** "Accuracy is necessary. It is not sufficient." + thank-you / Q&A.

## Design requirements
- 16:9, one slide per viewport, arrow-key + on-screen nav, slide counter.
- Dark theme, single accent colour, large type, minimal text per slide.
- Figures shown large and clean; captions small.
- Make the comparison table (slide 13) the visual centrepiece.
- Output a single `.html` file I can open in a browser and present.
