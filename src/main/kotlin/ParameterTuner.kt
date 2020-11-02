import java.lang.Double.max

class ParameterTuner(val evaluator: ModelEvaluator<*>, val data: DataFrame, val target: DataFrame, val runsPerVar: Int = 1) {
    private var curBestRes = 1e9
    private var bestParams: Params = Params()

    fun findBest(vararg paramVars: ParamVars): Params {
        val lparamVars = paramVars.toList()

        val params = Params()
        curBestRes = 1e9
        findBestRec(params, lparamVars)
        return bestParams
    }

    private fun findBestRec(params: Params, paramVars: List<ParamVars>) {
        if (paramVars.isEmpty()) {
            evaluator.setModelParams(params)
            params.print()
            var maxq = 0.0
            (0 until runsPerVar).forEach { i ->
                val q = evaluator.getDoubleQuality(data, target)
                maxq = max(maxq, q)
            }
            println("Score: $maxq")
            println("Current best result: $curBestRes")
            if (maxq < curBestRes) {
                curBestRes = maxq
                bestParams = params
            }
        } else {
            for (pval in paramVars[0].vars) {
                val newParams = params.copy()
                newParams.addParam(paramVars[0].paramName, pval)
                findBestRec(newParams, paramVars.drop(1))
            }
        }
    }
}