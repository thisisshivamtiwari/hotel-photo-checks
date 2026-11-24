# STUDY: hotel-photo-checks

Study notes: what was built, why it was built that way, what the numbers say, where it breaks, and the questions
someone reviewing it is likely to ask. Numbers are from results/summary.json (test set: 120 photos, 3 made-up hotels).

## 1. The problem in one paragraph
Listing photos are uploaded in batches. Some are blurry, dark, blown out or crooked, some are the same shot twice,
and some show an amenity the hotel doesn't have. An automated check is useful only if its "ok" can be trusted. It
has to be able to say "I can't tell", and it must never turn a blank fact or an unreadable file into a pass.

## 2. Design choices and why

| Choice | Why | Alternative considered |
|---|---|---|
| Real open-licence photos + edits with known defects | Labels for blur/exposure/tilt/duplicates are exact by construction; the photos are real scenes, not renders | Generated images (no image API key on this machine); real defective uploads (no licensed source) |
| No hotel photos, no business names in titles | Public repo; no real hotel can be named or shown | Commons hotel-room photos (dropped) |
| Three answers per check (ok / problem / unknown) | A wrong "ok" is the costly error; "unknown" sends a photo to a person instead of guessing | Binary classifier with a threshold |
| ±25% band around each rule threshold = unknown | Readings near the cut-off are coin flips; the band is fixed, not tuned | Tuned band (more test-set leakage risk) |
| Thresholds tuned on one dev hotel only | Every reported number comes from photos no threshold saw | Cross-validation (too few photos to be worth it) |
| Tilt from the median slant of near-vertical lines | A rotation slants every vertical the same way; looking up or down slants left and right ones in opposite directions, which cancels in the median | Horizon detection (fails indoors), a learned model |
| Fewer than 4 vertical lines = no tilt reading | Beaches, blurred photos and close-ups have no verticals; guessing there would be noise | Defaulting to "ok" (the trap the project is about) |
| CLIP zero-shot for amenities in the cheap arm | Runs on a laptop CPU in 0.03 s a photo, no training data needed | A fine-tuned detector (needs labelled boxes) |
| The LLM never sees the fact sheet | It reports what it sees; code compares with the facts, so a persuasive fact sheet can't bias what it "sees" | Ask the LLM "does this contradict the facts?" |
| Blank fact = "unverifiable", never yes or no | A blank is missing data, not evidence; scoring it as a pass is exactly the "missing data scores 100" bug | Treat blank as yes (would confirm 9 claims here) |
| Merge: agree → answer, one unknown → the other, disagree → unknown | Uses the second opinion where one side is blind; disagreements go to a person | Strict merge (added post hoc, reported separately) |
| pHash for duplicates, rules only | Pairwise; an LLM per pair would be N² calls for a job hashing does well | CLIP embedding similarity |
| Recorded replies (cassettes) + saved CLIP scores | CI reproduces every number with no key and no torch | Live calls in CI (cost, flakiness) |
| Gate written before the first test run | Stops the pass bar being fitted to the result; the one broken rule stays visible | Writing the gate after seeing numbers |

## 3. Results (test set)

| Checker | Bad passed as ready (of 75) | Good ready (of 45) | Good sent to a person | Contradictions caught (of 20) | False contradictions | $/photo |
|---|---|---|---|---|---|---|
| Rules + CLIP | 2 | 22 | 19 | 17 | 1 | 0 |
| Haiku 4.5 | 35 | 44 | 0 | 17 | 1 | 0.0047 |
| Sonnet 5.5 | 14 | 41 | 1 | 20 | 0 | 0.0063 |
| Rules + Haiku | 8 | 40 | 5 | 15 | 0 | 0.0047 |
| Rules + Sonnet | 4 | 37 | 6 | 18 | 0 | 0.0063 |
| Strict Rules + Sonnet (post hoc) | 1 | 21 | 24 | 17 | 0 | 0.0063 |

- Sonnet alone vs Rules + Sonnet, bad photos passed: 10 vs 0 discordant, p = 0.002 (exact McNemar).
- Rules alone vs Rules + Sonnet: 1 vs 3, p = 0.625, so the pre-registered rule "combined passes fewer bad photos
  than the rules" failed, but the difference isn't significant either way.
- Every arm caught all 30 blurred photos. The differences are in exposure and tilt.
- Duplicates (pHash): 6/15 caught, 8 "possible", 1 missed, 0 false among 2,160 pairs of different shots.

## 4. Failure modes seen
1. **LLMs pass over-exposed photos.** Sonnet passed 8 of 15 bright copies as ready, often while its note says the
   window is "blown out". The note saw it but the field said ok.
2. **Haiku passes dark photos.** It caught 8 of 30 exposure defects. The prompt line "a moody but readable photo is
   ok" plausibly gives it permission (4 notes say "moody").
3. **LLMs under-call tilt.** Sonnet caught 8 of 15 and Haiku 1 of 15, while Haiku also raised 17 false tilt alarms
   on 105 photos without one.
4. **Rules can't see tilt without straight lines.** Many photos get no reading (beaches, blurred copies), so the
   rules alone send many good photos to a person.
5. **The merge's weak spot.** When the rules can't tell and the LLM says "ok", the merge passes it. That happened on
   p11-tilt and p30-tilt, two of the four bad photos Rules + Sonnet passed.
6. **CLIP misses fireplaces in busy rooms.** It missed 7 and saw 1 that wasn't there, mostly dining rooms where the
   "dining room" prompt wins the softmax.
7. **Edits as labels.** p23-bright (over-exposure factor 3 on a dim room) has almost no blown pixels. Both checkers
   passed it, and the label may simply be harsh.
8. **Notes carry account text.** Many Sonnet notes end with "please verify before use", added by Claude Code account
   instructions. The fields are unaffected.

## 5. Ten likely questions, with answers

1. **Why not train a classifier?** 40 source photos is far too few to train and also test on. Zero-shot models and
   classic measures need no training, and the edits give exact labels to score them. With a few thousand real
   uploads, a small fine-tuned model per defect would be the next step.
2. **Isn't the test set too easy, since the defects are synthetic?** Partly. Blur turned out easy for everyone. Exposure and tilt
   weren't, and those are where the checkers differ. Real uploads mix defects and have subtler ones, so treat these
   numbers as a ranking of approaches, not production accuracy.
3. **Why is "unknown" good?** Because the errors aren't symmetric. A wrong "ok" publishes a bad photo, a wrong
   "unknown" costs a person a few seconds. The design goal is few wrong "ok"s at a workable unknown rate: Rules +
   Sonnet sends 6 of 45 good photos and 12 of 75 bad ones to a person.
4. **How did you avoid tuning on the test set?** Thresholds were fitted on one hotel and frozen in
   results/thresholds.json. The gate was committed before any test result existed. The one change made after
   seeing results (the strict merge) is labelled post hoc, and the original merge's numbers are still reported.
5. **Why does the combined checker pass more bad photos than the rules alone?** The merge takes the LLM's answer
   when the rules can't tell. That covers most of the rules' unknowns correctly (good photos ready: 22 → 37), but it
   let two tilted photos through. Whether 2 extra bad photos are worth 15 extra good ones is a product decision.
   The strict merge shows the other end of the trade-off.
6. **Why doesn't the LLM see the fact sheet?** So its answer is about the photo. Given "the hotel has no pool", a
   model may talk itself out of seeing one. The comparison is plain code, and is tested.
7. **How do you handle broken uploads?** A gate runs before any check: missing, empty, cut-off, non-image and
   under-200 px files become "unreadable" with a reason, and no model is called. CMYK, 16-bit, alpha, WebP and EXIF
   rotation are converted. Only the JPEG, PNG and WebP decoders are ever tried: Pillow picks a decoder from the first
   bytes, so a PostScript file named .jpg would otherwise go to Ghostscript; now it is "unreadable". Tests cover each
   case, including that one. A failed or unparseable LLM reply becomes "unknown" for every field.
8. **What does it cost?** Rules + CLIP are free and take 0.04 s a photo on a laptop CPU. Sonnet 5.5 at low effort is
   $0.0063 a photo at list prices through the CLI (so including its overhead). A 40-photo batch costs about 25 cents.
9. **Why pHash and not embeddings for duplicates?** pHash caught or flagged 14 of 15 re-exports with no false
   matches, even on two shots of the same room. Embeddings would rate those two shots as very similar, which is
   the wrong answer here.
10. **What would you change first?** Ask the LLM for a severity score and let it overrule the rules only inside their
    unknown band, test the prompt without the "moody" line, and add a second model family for amenity calls.

## 6. Reproduce
`uv sync && uv run python -m photo_checks ci` replays everything and checks it matches results/summary.json. To
re-record, see README "Run it".
