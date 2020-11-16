import java.lang.Double.max
import java.lang.Double.min

class ParameterTuner(val evaluator: ModelEvaluator<*>, val data: DataFrame, val target: DataFrame, val runsPerVar: Int = 1) {
    private var curBestRes = 1e9
    private var bestParams: Params = Params()

    fun findBest(minimize: Boolean, vararg paramVars: ParamVars): Params {
        val lparamVars = paramVars.toList()

        val params = Params()
        curBestRes = if (minimize) 1e9 else -1e9
        findBestRec(minimize, params, lparamVars)
        return bestParams
    }

    fun findBest(vararg paramVars: ParamVars): Params {
        return findBest(true, *paramVars)
    }

    private fun findBestRec(minimize: Boolean, params: Params, paramVars: List<ParamVars>) {
        if (paramVars.isEmpty()) {
            evaluator.setModelParams(params)
            params.print()
            var maxq = if (minimize) -1e9 else 1e9
            (0 until runsPerVar).forEach { i ->
                val q = evaluator.getDoubleQuality(data, target)
                print("$q ")
                maxq = if (minimize) max(maxq, q) else min(maxq, q)
            }
            println("Score: $maxq")
            println("Current best result: $curBestRes")
            if (minimize) {
                if (maxq < curBestRes) {
                    curBestRes = maxq
                    bestParams = params
                }
            } else {
                if (maxq > curBestRes) {
                    curBestRes = maxq
                    bestParams = params
                }
            }
        } else {
            for (pval in paramVars[0].vars) {
                val newParams = params.copy()
                newParams.addParam(paramVars[0].paramName, pval)
                findBestRec(minimize, newParams, paramVars.drop(1))
            }
        }
    }
}