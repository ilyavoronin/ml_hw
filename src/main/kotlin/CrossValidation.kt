import java.lang.Integer.min

class CrossValidation(override val model: MLModel) : ModelEvaluator<List<Double>> {
    private var myK: Int = 0
    private var mySeed: Long? = null

    fun setSeed(seed: Long) {
        mySeed = seed
    }

    fun withK(k: Int): CrossValidation {
        myK = k
        return this
    }
    override fun getQuality(data: DataFrame, target: DataFrame): List<Double> {
        if (myK == 0) {
            myK = data.rowsCnt()
        }

        val ress = mutableListOf<Double>()

        val (shuffledData, shuffledTarget) = data.shuffleWith(target, mySeed)

        val testSize = data.rowsCnt() / myK

        for (i in 0 until shuffledData.rowsCnt() step testSize) {
            print(i.toString() + " ")
            val l = i
            val r = min(shuffledData.rowsCnt(), i + testSize)

            val res = shuffledData.extractWith(shuffledTarget, l, r)

            val curRes = runModel(model, res.part1, res.part1other, res.part2, res.part2other)
            ress.add(curRes)
        }
        println()
        return ress
    }

    override fun getDoubleQuality(data: DataFrame, target: DataFrame): Double {
        val q = getQuality(data, target)
        return q.max()!!
    }

    private fun runModel(model: MLModel, data: DataFrame, target: DataFrame, testData: DataFrame, testTarget: DataFrame): Double {
        model.fit(data, target)

        val res = model.predict(testData)

        return meanSqError(res, testTarget)
    }

}