import kotlin.math.pow

fun main(args: Array<String>) {
    while (true) {
        print("Enter binary decimal number: ")
        val input = readln()
        if (input == "exit") {
            break
        }
        print("Binary output: ")
        println(bcdToBinary(input))
        println("===========")
    }
}

fun bcdToBinary(s: String): String {
    val regex = "^[0-1]+\$"
    if (!Regex(regex).matches(s))
        return "Invalid input"
    val l = s.length
    var num: Int
    var mul = 1
    var sum = 0

    // If the length of given BCD is not
    // divisible by 4
    for (i in l % 4 - 1 downTo 0) {
        sum += (s[i].code - '0'.code) * mul
        mul *= 2
    }
    num = sum
    sum = 0
    mul = 2.0.pow(3.0).toInt()
    var ctr = 0
    for (i in l % 4 until l) {
        ctr++
        sum += (s[i].code - '0'.code) * mul
        mul /= 2
        if (ctr == 4) {
            num = num * 10 + sum
            sum = 0
            mul = 2.0.pow(3.0).toInt()
            ctr = 0
        }
    }

    // Convert decimal to binary
    var ans = ""
    while (num > 0) {
        ans += (num % 2 + '0'.code).toChar()
        num /= 2
    }
    ans = ans.reversed()
    return ans
}