# Geometry Benchmark Analysis Summary

- Problems analyzed: **30**
- Construction/definition templates: **72**
- Inference rules parsed: **43**

## Dataset complexity
- Premise clauses per problem: min=4, median=8.0, mean=8.67, max=15
- Constraint atoms per problem: min=5, median=12.0, mean=12.50, max=19
- Unique primitives per problem: min=3, median=5.5, mean=5.50, max=8
- Goal arity: min=3, median=4.0, mean=4.17, max=8

## Goal predicates
- cong: 12
- coll: 7
- cyclic: 5
- eqangle: 2
- perp: 2
- eqratio: 1
- para: 1

## Most frequent construction primitives
- on_line: 120
- on_circle: 72
- circle: 28
- triangle: 23
- on_bline: 18
- on_tline: 17
- midpoint: 14
- on_aline: 13
- foot: 11
- reflect: 10
- angle_bisector: 8
- mirror: 6
- segment: 5
- on_pline: 5
- orthocenter: 5

## Most frequent rule consequents
- eqangle: 7
- para: 6
- cong: 5
- perp: 5
- contri*: 3
- cyclic: 2
- eqratio: 2
- eqratio6: 2
- midp: 2
- simtri*: 2
- eqratio3: 1
- eqangle6: 1

## Most structurally complex benchmark items
- translated_imo_2011_p6: 15 clauses, 19 atoms, goal=coll
- translated_imo_2008_p6: 13 clauses, 18 atoms, goal=cong
- translated_imo_2008_p1a: 11 clauses, 17 atoms, goal=cyclic
- translated_imo_2008_p1b: 11 clauses, 17 atoms, goal=cyclic
- translated_imo_2015_p4: 11 clauses, 17 atoms, goal=coll

## Least structurally complex benchmark items
- translated_imo_2004_p5: 4 clauses, 5 atoms, goal=cong
- translated_imo_2012_p5: 6 clauses, 9 atoms, goal=cong
- translated_imo_2018_p1: 6 clauses, 9 atoms, goal=para
- translated_imo_2009_p2: 8 clauses, 9 atoms, goal=cong
- translated_imo_2012_p1: 6 clauses, 10 atoms, goal=cong

## Interpretation
The benchmark combines a compact set of high-level geometric construction primitives with a rule base that rewrites between incidence, angle, ratio, congruence, cyclicity, and parallel/perpendicular relations. This structure is suitable for a neuro-symbolic theorem prover that first predicts useful intermediate predicates or construction expansions, then validates them through symbolic rule chaining. The analysis artifacts in outputs/ and report/images/ are intended to support that later modeling/reporting stage.
