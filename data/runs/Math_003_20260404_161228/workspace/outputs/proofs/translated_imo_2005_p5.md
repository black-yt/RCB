# translated_imo_2005_p5
## Formal goal
- `cyclic p q r m`
## Proof plan
1. Parse the construction into 10 clauses over 17 symbolic points.
2. Ground constructor semantics for `circle, eqdistance` to expose implicit incidences, equalities, and perpendicular/parallel relations.
3. Goal-directed strategy: establish equal angles or equal powers to invoke a cyclicity rule.
4. Highest-scoring reusable lemmas/rules:
   1. [score=2] `cong O A O B, cong O B O C, cong O C O D => cyclic A B C D`
   2. [score=2] `eqangle6 P A P B Q A Q B, ncoll P Q A B => cyclic A B P Q`
   3. [score=1] `cyclic A B P Q => eqangle P A P B Q A Q B`
   4. [score=1] `cyclic A B C P Q R, eqangle C A C B R P R Q => cong A B P Q`
   5. [score=1] `circle O A B C, perp O A A X => eqangle A X A B C A C B`
5. Attempt a machine-verifiable derivation by instantiating the retrieved rules on symbols introduced in the construction; unresolved variable bindings are left explicit for downstream search.
