package ru.kheynov.hotel.data.mappers

import ru.kheynov.hotel.data.entities.RefreshToken
import ru.kheynov.hotel.domain.entities.UserDTO

fun RefreshToken.toRefreshTokenInfo(): UserDTO.RefreshTokenInfo {
    return UserDTO.RefreshTokenInfo(
        userId = this.userId,
        clientId = this.clientId,
        token = this.refreshToken,
        expiresAt = this.expiresAt,
    )
}

fun ru.kheynov.hotel.jwt.token.RefreshToken.toDataRefreshToken(
    userId: String,
    clientId: String,
): RefreshToken {
    val refreshToken = this
    return RefreshToken {
        this.userId = userId
        this.clientId = clientId
        this.refreshToken = refreshToken.token
        this.expiresAt = refreshToken.expiresAt
    }
}