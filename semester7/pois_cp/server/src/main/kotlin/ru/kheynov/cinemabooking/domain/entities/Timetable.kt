package ru.kheynov.cinemabooking.domain.entities

import java.time.LocalDate

data class Timetable(
    val id: String,
    val cinemaId: String,
    val filmId: String,
    val time: LocalDate,
)
