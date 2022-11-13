# Notes

## Story

we want to see pro1/pro2/neutral first, so push docs with no stance to end of ranking
-> improves ndcg@5 for all touché runs
but now we might have pushed subjective documents to the top, so let's ensure rankings are fair now
discount scores by the group exposure in the ranking

Important: Protected groups are calculated per query because pro1 doesn't describe the same objects in different queries.
