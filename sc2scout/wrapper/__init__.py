from sc2scout.wrapper.wrapper_factory import make, register, model
from sc2scout.wrapper.explore_enemy import ExploreMakerV0, \
ExploreMakerV2, ExploreMakerV6, ExploreMakerV8, ExploreMakerV9, \
ExploreMakerV10, ExploreMakerV12
from sc2scout.wrapper.evade_enemy import EvadeMakerV0, EvadeMakerV1
from sc2scout.wrapper.explore_target import TargetMakerV1
from sc2scout.wrapper.ground_explore import ZerglingEvadeMakerLocalV0, \
ZerglingEvadeMakerGlobalV0, GroundMakerV2, GroundMakerV3, GroundMakerV4

register('explore_v0', ExploreMakerV0())
register('explore_v2', ExploreMakerV2())
register('explore_v6', ExploreMakerV6())
register('explore_v8', ExploreMakerV8())
register('explore_v9', ExploreMakerV9())
register('explore_v10', ExploreMakerV10())
register('explore_v12', ExploreMakerV12())
register('evade_v0', EvadeMakerV0())
register('evade_v1', EvadeMakerV1())
register('target_v1', TargetMakerV1())
register('ground_v0', ZerglingEvadeMakerLocalV0())
register('ground_v1', ZerglingEvadeMakerGlobalV0())
register('ground_v2', GroundMakerV2())
register('ground_v3', GroundMakerV3())
register('ground_v4', GroundMakerV4())

