model = nn.Sequential(
    StiefelLinear(256, 128),
    nn.ReLU(),
    StiefelLinear(128, 64),
)

opt = StiefelOptimizer(model.parameters(),
                       lr=1e-3,
                       momentum=0.9,
                       retraction="cayley",
                       invariant_mode="warn")
