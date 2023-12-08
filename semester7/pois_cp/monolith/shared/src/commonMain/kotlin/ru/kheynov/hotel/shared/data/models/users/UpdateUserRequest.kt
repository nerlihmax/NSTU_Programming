package ru.kheynov.hotel.shared.data.models.users

import kotlinx.serialization.Serializable

@Serializable
data class UpdateUserRequest(
    val username: String,
)
