package ru.kheynov.hotel.shared.data.storage

import ru.kheynov.hotel.shared.jwt.TokenPair
import java.io.File

actual class TokenStorage(
    private val file: File,
) {
    actual fun saveToken(token: TokenPair) {
        file.printWriter().use { out ->
            out.println(token.accessToken)
            out.println(token.refreshToken.token)
            out.println(token.refreshToken.expiresAt.toString())
        }
    }

    actual fun getRefreshToken(): String? {
        return file.readLines().getOrNull(1)
    }

    actual fun clearToken() {
        file.writeText("")
    }
}