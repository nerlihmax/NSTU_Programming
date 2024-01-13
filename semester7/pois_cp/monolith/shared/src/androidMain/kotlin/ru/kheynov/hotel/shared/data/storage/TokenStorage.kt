package ru.kheynov.hotel.shared.data.storage

import android.content.SharedPreferences
import ru.kheynov.hotel.shared.jwt.TokenPair

actual class TokenStorage(
    private val prefs: SharedPreferences
) {
    actual fun saveToken(token: TokenPair) {
        prefs.edit()
            .putString("access_token", token.accessToken)
            .putString("refresh_token", token.refreshToken.token)
            .putString("refresh_token_exp", token.refreshToken.expiresAt.toString())
            .apply()
    }

    actual fun getRefreshToken(): String? {
        return prefs.getString("refresh_token", null)
    }

    actual fun clearToken() {
        prefs.edit()
            .remove("access_token")
            .remove("refresh_token")
            .remove("refresh_token_exp")
            .apply()
    }
}