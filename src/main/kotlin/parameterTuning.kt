
fun main() {
    val cv = CrossValidation(LinearRegression()).withK(20)
    val alldata = DataLoader("data/comp1_train.csv").read()

    val (target, data) = alldata.splitColsBy(1)

    val tuner = ParameterTuner(cv, data, target, 3)

    val bestParams = tuner.findBest(
        ParamVars("gradRed", 0.093, 0.094, 0.0935, 0.0925),
        ParamVars("stepPow", 1.95, 1.96, 1.965, 1.955),
        ParamVars("algConvEps", 0.1),
        ParamVars("initialStepSize", 7.3, 7.4, 7.45, 7.5, 7.55, 7.6),
        ParamVars("lam", 0.1, 0.15, 0.08, 0.12)
    )
    bestParams.print()
}