package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable
import ru.kheynov.hotel.shared.utils.LocalDateSerializer
import java.time.LocalDate

@Serializable
data class DateRange(
    @Serializable(with = LocalDateSerializer::class)
    val from: LocalDate,

    @Serializable(with = LocalDateSerializer::class)
    val to: LocalDate
)