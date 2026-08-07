"""Short-term, episodic, and semantic memory for the Swiftrail agent.

This package is deliberately split into layers that never write past
their own boundary:

    ShortTermBuffer / Scratchpad  -->  PromoteDropRouter  -->  EpisodicMemory
                                                                     |
                                                          ConsolidationLayer (periodic)
                                                                     |
                                                              SemanticMemory

The router only ever decides "forget" or "episodic". It never writes to
SemanticMemory directly. Semantic facts are only ever produced by a
separate, periodic ConsolidationLayer pass over EpisodicMemory.
"""
