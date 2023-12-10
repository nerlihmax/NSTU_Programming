package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable
import ru.kheynov.hotel.shared.utils.LocalDateSerializer
import java.time.LocalDate

@Serializable
data class Room(
    val id: String,
    val type: String,
    val price: Int,
    val hotel: Hotel,
)

@Serializable
data class RoomInfo(
    val id: String,
    val type: String,
    val price: Int,
)

@Serializable
data class RoomReservation(
    val id: String,
    val user: User,
    val room: Room,
    @Serializable(with = LocalDateSerializer::class) val from: LocalDate,
    @Serializable(with = LocalDateSerializer::class) val to: LocalDate
)

@Serializable
data class RoomReservationInfo(
    val id: String,
    val room: Room,
    val user: User? = null,
    @Serializable(with = LocalDateSerializer::class) val from: LocalDate,
    @Serializable(with = LocalDateSerializer::class) val to: LocalDate,
)
