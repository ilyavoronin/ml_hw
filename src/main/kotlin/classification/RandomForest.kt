package classification

import DataFrame
import MLModel
import Params
import java.util.*

class RandomForest(
        var numberOfTrees: Int,
        var maxTreeDepth: Int,
        var minNodeObjects: Int,
        var cntParamsToUse: Int
): MLModel {
    private var myRandom = Random()
    private var myTrees = mutableListOf<DecisionTreeClassifier>()

    constructor(vararg decisionTrees: DecisionTreeClassifier): this(0, 0, 0, 0) {
        myTrees = decisionTrees.toMutableList()
    }

    fun setSeed(seed: Long) {
        myRandom = Random(seed)
    }

    override fun setParams(params: Params) {
        TODO("Not yet implemented")
    }

    override fun fit(table: DataFrame, target: DataFrame) {
        if (myTrees.isNotEmpty()) {
            myTrees.forEach {tree ->
                tree.fit(table, target)
            }
            return
        }

        (0 until numberOfTrees).forEach {
            val newData = table.likeC()
            val newTarget = target.likeC()

            (0 until table.rowsCnt()).forEach {
                val i = myRandom.nextInt(table.rowsCnt())
                newData.appendRow(table[i])
                newTarget.appendRow(target[i])
            }

            val paramsToUse = (0 until table.colsCnt()).toList().shuffled(myRandom).take(cntParamsToUse)

            val newTree = DecisionTreeClassifier(maxTreeDepth, minNodeObjects)
            newTree.fit(newData, newTarget)

            myTrees.add(newTree)
        }
    }

    override fun predict(table: DataFrame): DataFrame {
        val ress = mutableListOf<DataFrame>()

        myTrees.forEach {tree ->
            ress.add(tree.predict(table))
        }

        val res = DataFrame(0, 1)

        (0 until table.rowsCnt()).forEach {i ->
            val k1 = ress.sumByDouble { df -> df[i][0] } / myTrees.size

            res.appendRow(listOf(k1))
        }

        return res
    }
}