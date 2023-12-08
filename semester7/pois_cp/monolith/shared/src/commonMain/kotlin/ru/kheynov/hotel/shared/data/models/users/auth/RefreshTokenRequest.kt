package ru.kheynov.hotel.shared.data.models.users.auth

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RefreshTokenRequest(
    @SerialName("refresh_token") val oldRefreshToken: String,
)