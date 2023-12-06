package ru.kheynov.hotel.domain.entities

import java.time.LocalDateTime

data class Reservation(
    val id: Int,
    val guest: User,
    val arrivalDate: LocalDateTime,
    val departureDate: LocalDateTime,
    val room: Room,
)
