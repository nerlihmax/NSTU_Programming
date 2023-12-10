package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable

@Serializable
data class User(
    val id: String,
    val name: String,
    val email: String,
)

@Serializable
data class UserUpdate(
    val name: String? = null,
    val email: String? = null,
)

@Serializable
data class RefreshTokenInfo(
    val userId: String,
    val clientId: String,
    val token: String,
    val expiresAt: Long,
)

@Serializable
data class UserInfo(
    val id: String,
    val name: String,
    val email: String,
    val passwordHash: String,
)

@Serializable
data class UserEmailSignUp(
    val name: String,
    val password: String,
    val email: String,
    val clientId: String,
)