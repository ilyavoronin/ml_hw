class Params {
    private val myMap = mutableMapOf<String, Double>()
    fun addParam(paramName: String, paramValue: Double) {
        myMap.put(paramName, paramValue)
    }

    fun getParam(paramName: String): Double? {
        return myMap.get(paramName)
    }

    fun copy(): Params {
        val newParams = Params()
        myMap.forEach {k, v ->
            newParams.addParam(k, v)
        }
        return newParams
    }

    fun print() {
        myMap.forEach {k, v ->
            println("$k: $v")
        }
    }
}

class ParamVars(val paramName: String, vararg val vars: Double)