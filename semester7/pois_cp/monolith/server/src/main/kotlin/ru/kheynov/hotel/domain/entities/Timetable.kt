package ru.kheynov.hotel.domain.entities

import java.time.LocalDate

data class Timetable(
    val id: String,
    val cinemaId: String,
    val filmId: String,
    val time: LocalDate,
)
