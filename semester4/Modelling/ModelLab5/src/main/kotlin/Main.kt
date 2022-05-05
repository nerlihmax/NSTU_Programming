import java.lang.Math.random

private const val a = 1
private const val b = 2
private const val SIZE = 400
private const val w =b

fun f(x: Double): Double = x - 0.5

fun main() {
    val kx = Array(SIZE) { random() }
    val ky = Array(SIZE) { random() }
    for (i in 0 until SIZE) {
        val x = a + kx[i] * (b - a)
        val y = w * ky[i]
        if (y <= f(x)) println(x)
    }
}

