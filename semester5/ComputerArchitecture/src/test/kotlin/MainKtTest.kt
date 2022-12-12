import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

internal class MainKtTest {

    @Test
    fun `BCD to binary`() {
        assertEquals("10000111011111", bcdToBinary("1000011001110001"))
        assertEquals("1110001111", bcdToBinary("100100010001"))
        assertEquals("11101", bcdToBinary("101001"))
    }
}
