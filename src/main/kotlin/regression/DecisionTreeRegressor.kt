package regression

import DataFrame
import MLModel
import Params
import kotlin.math.pow

class DecisionTreeRegressor(val maxDepth: Int, val minNodeData: Int) : MLModel {

    class Node {
        var isTerminal: Boolean = false
        var featureIndex: Int? = null
        var bound: Double? = null
        var left: Node? = null
        var right: Node? = null
        var ans: Double? = null
    }

    private var myRoot: Node? = null
    override fun setParams(params: Params) {
        TODO("Not yet implemented")
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
        (0 until table.colsCnt()).forEach {j ->
            (0 until table.rowsCnt()).forEach { medInd ->
                val med = table[medInd][j]
                val leftT = DataFrame(0, 1)
                val rightT = DataFrame(0, 1)
                (0 until table.rowsCnt()).forEach { i ->
                    if (table[i, j] <= med) {
                        leftT.appendRow(target[i])
                    } else {
                        rightT.appendRow(target[i])
                    }
                }
                if (leftT.rowsCnt() >= minNodeData && rightT.rowsCnt() >= minNodeData) {
                    val curGain = unc - 0.5 * caclulateUncertanty(leftT) - 0.5 * caclulateUncertanty(rightT)
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
    }

    private fun shouldStop(table: DataFrame, target: DataFrame, depth: Int): Boolean {
        return table.rowsCnt() < 2 * minNodeData || depth >= maxDepth || table.rowsCnt() == 1
    }

    private fun getAns(target: DataFrame): Double {
        return target.tr()[0].sum() / target.rowsCnt()
    }

    private fun caclulateUncertanty(target: DataFrame): Double {
        var res = 1e18
        val t = target.tr().get(0)

        t.forEachIndexed{i, y ->
            var curRes = 0.0
            t.forEach {y_i ->
                curRes += (y - y_i).pow(2)
            }
            curRes /= t.size
            if (curRes < res) {
                res = curRes
            }
        }
        return res
    }

    private fun getAns(node: Node, row: List<Double>): Double {
        if (node.isTerminal) {
            return node.ans!!
        }

        if (row[node.featureIndex!!] <= node.bound!!) {
            return getAns(node.left!!, row)
        } else {
            return getAns(node.right!!, row)
        }
    }
}