package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable
import ru.kheynov.hotel.shared.utils.LocalDateSerializer
import java.time.LocalDate

@Serializable
data class Reservation(
    val id: Int,
    val guest: User,
    @Serializable(with = LocalDateSerializer::class) val arrivalDate: LocalDate,
    @Serializable(with = LocalDateSerializer::class) val departureDate: LocalDate,
    val room: Room,
)
