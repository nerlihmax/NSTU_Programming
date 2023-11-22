package ru.kheynov.hotel.data.models.users.auth

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RefreshTokenRequest(
    @SerialName("refresh_token") val oldRefreshToken: String,
)