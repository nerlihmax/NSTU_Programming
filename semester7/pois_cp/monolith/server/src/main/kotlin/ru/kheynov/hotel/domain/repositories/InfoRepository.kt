package ru.kheynov.cinemabooking.domain.repositories

import ru.kheynov.cinemabooking.domain.entities.Cinema
import ru.kheynov.cinemabooking.domain.entities.Film
import ru.kheynov.cinemabooking.domain.entities.Timetable
import java.time.LocalDate

interface InfoRepository {
    suspend fun getAvailableCinemas(): List<Cinema>
    suspend fun getAvailableFilms(): List<Film>
    suspend fun getTimetables(cinemaId: String, date: LocalDate): List<Timetable>
}