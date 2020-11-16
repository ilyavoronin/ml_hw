package classification

import DataFrame
import MLModel
import Params
import kotlin.math.log2
import kotlin.math.pow

class DecisionTreeClassifier(
        var maxDepth: Int,
        var minNodeData: Int,
        var parametersToUse: List<Int>? = null,
        val uncertantyFunction: Uncertanty = Uncertanty.ENT
) : MLModel {

    enum class Uncertanty(val f: (target :DataFrame) -> Double) {
        ENT({target: DataFrame ->
            val t = target.tr().get(0)
            val q = (t.count {it == 1.0}).toDouble() / t.size
            -q * log2(q) - (1 - q) * log2(1 - q)
        }),
        GINI({target ->
            val t = target.tr().get(0)
            val q = (t.count {it == 1.0}).toDouble() / t.size
            2.0 * q * (1 - q)
        }),
        SQUARED({target ->
            val t = target.tr().get(0)
            val q = (t.count {it == 1.0}).toDouble() / t.size
            q * (1 - q).pow(2) + (1 - q) * q.pow(2)
        }),
        G05({target ->
            val t = target.tr().get(0)
            val q = (t.count {it == 1.0}).toDouble() / t.size
            q * (1 - q).pow(0.8) + (1 - q) * q.pow(0.8)
        }),
        G15({target ->
            val t = target.tr().get(0)
            val q = (t.count {it == 1.0}).toDouble() / t.size
            q * (1 - q).pow(1.5) + (1 - q) * q.pow(1.5)
        })
    }

    constructor(): this(10, 4)

    class Node {
        var isTerminal: Boolean = false
        var featureIndex: Int? = null
        var bound: Double? = null
        var left: Node? = null
        var right: Node? = null
        var ans: Double? = null
        var leftCoef: Double? = null
    }

    private var myRoot: Node? = null

    override fun setParams(params: Params) {
        params.getParam("maxDepth")?.let { maxDepth = it.toInt() }
        params.getParam("minNodeData")?.let { minNodeData = it.toInt() }
    }


    override fun fit(table: DataFrame, target: DataFrame) {
        val root = Node()
        buildTree(root, table, target, 0)
        myRoot = root
    }

    override fun predict(table: DataFrame): DataFrame {
        val res = DataFrame(0, 1)
        (0 until table.rowsCnt()).forEach {i ->
            res.appendRow(listOf(getAns(myRoot!!, table[i])))
        }
        return res
    }

    private fun buildTree(node: Node, table: DataFrame, target: DataFrame, depth: Int) {
        if (shouldStop(table, target, depth)) {
            node.isTerminal = true
            node.ans = getAns(target)
            return
        }
        val unc = caclulateUncertanty(target)
        var maxGain = 0.0
        var bestInd = -1
        var div = -1.0

        if (parametersToUse == null) {
            parametersToUse = (0 until table.colsCnt()).toList()
        }

        parametersToUse!!.forEach {j ->
            (0 until table.rowsCnt()).forEach { medInd ->
                val med = table[medInd][j]
                val allT = DataFrame(0, 1)
                val leftT = DataFrame(0, 1)
                val rightT = DataFrame(0, 1)
                (0 until table.rowsCnt()).forEach { i ->
                    if (!table[i, j].isNaN()) {
                        allT.appendRow(target[i])
                        if (table[i, j] <= med) {
                            leftT.appendRow(target[i])
                        } else {
                            rightT.appendRow(target[i])
                        }
                    }
                }

                if (leftT.rowsCnt() >= minNodeData && rightT.rowsCnt() >= minNodeData) {
                    val curGain = caclulateUncertanty(allT) -
                            (leftT.rowsCnt().toDouble() / table.rowsCnt()) * caclulateUncertanty(leftT) -
                            (rightT.rowsCnt().toDouble() / table.rowsCnt()) * caclulateUncertanty(rightT)

                    if (bestInd == -1 || curGain > maxGain) {
                        bestInd = j
                        maxGain = curGain
                        div = med
                    }
                }

            }
        }

        if (bestInd == -1) {
            node.isTerminal = true
            node.ans = getAns(target)
            return
        }

        val lnode = Node()
        val rnode = Node()

        val ldata = table.likeC()
        val rdata = table.likeC()
        val ltartget = target.likeC()
        val rtargert = target.likeC()
        (0 until table.rowsCnt()).forEach {i ->
            if (table[i, bestInd] <= div) {
                ldata.appendRow(table[i])
                ltartget.appendRow(target[i])
            } else {
                rdata.appendRow(table[i])
                rtargert.appendRow(target[i])
            }
        }

        buildTree(lnode, ldata, ltartget, depth + 1)
        buildTree(rnode, rdata, rtargert, depth + 1)
        node.left = lnode
        node.right = rnode
        node.featureIndex = bestInd
        node.bound = div
        node.leftCoef = ldata.rowsCnt().toDouble() / table.rowsCnt()
    }

    private fun shouldStop(table: DataFrame, target: DataFrame, depth: Int): Boolean {
        return table.rowsCnt() < 2 * minNodeData || depth >= maxDepth || table.rowsCnt() == 1
    }

    private fun getAns(target: DataFrame): Double {
        return target.tr().get(0).count {it == 1.0}.toDouble() /  target.rowsCnt()
    }

    private fun caclulateUncertanty(target: DataFrame): Double {
        return uncertantyFunction.f(target)
    }

    private fun getAns(node: Node, row: List<Double>): Double {
        if (node.isTerminal) {
            return node.ans!!
        }

        if (row[node.featureIndex!!].isNaN()) {
            return getAns(node.left!!, row) * node.leftCoef!! + getAns(node.right!!, row) * (1 - node.leftCoef!!)
        }

        if (row[node.featureIndex!!] <= node.bound!!) {
            return getAns(node.left!!, row)
        } else {
            return getAns(node.right!!, row)
        }
    }
}