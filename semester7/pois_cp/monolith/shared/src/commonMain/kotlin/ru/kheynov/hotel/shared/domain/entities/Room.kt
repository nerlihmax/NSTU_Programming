package ru.kheynov.hotel.shared.domain.entities

import java.time.LocalDate

data class Room(
    val id: String,
    val type: String,
    val price: Int,
    val hotel: Hotel,
)

data class RoomReservation(
    val id: String,
    val user: User,
    val room: Room,
    val from: LocalDate,
    val to: LocalDate
)

data class RoomReservationInfo(
    val id: String,
    val room: Room,
    val from: LocalDate,
    val to: LocalDate,
)
