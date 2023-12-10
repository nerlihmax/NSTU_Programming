package ru.kheynov.hotel.shared.data.models.reservations

import kotlinx.serialization.Serializable

@Serializable
data class ReserveRoomRequest(
    val roomId: String,
    val from: String,
    val to: String,
)