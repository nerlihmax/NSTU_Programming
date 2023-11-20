package ru.kheynov.cinemabooking.api.requests.users.auth

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RefreshTokenRequest(
    @SerialName("refresh_token") val oldRefreshToken: String,
)