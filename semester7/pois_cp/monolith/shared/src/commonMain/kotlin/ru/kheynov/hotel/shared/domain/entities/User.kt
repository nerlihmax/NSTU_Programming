package ru.kheynov.hotel.shared.domain.entities

data class User(
    val id: String,
    val name: String,
    val email: String,
)

data class UserUpdate(
    val name: String? = null,
    val email: String? = null,
)

data class RefreshTokenInfo(
    val userId: String,
    val clientId: String,
    val token: String,
    val expiresAt: Long,
)

data class UserInfo(
    val id: String,
    val name: String,
    val email: String,
    val passwordHash: String,
)

data class UserEmailSignUp(
    val name: String,
    val password: String,
    val email: String,
    val clientId: String,
)