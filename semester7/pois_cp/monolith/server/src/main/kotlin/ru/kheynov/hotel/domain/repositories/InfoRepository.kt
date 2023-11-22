package ru.kheynov.hotel.domain.repositories

import ru.kheynov.hotel.domain.entities.Cinema
import ru.kheynov.hotel.domain.entities.Film
import ru.kheynov.hotel.domain.entities.Timetable
import java.time.LocalDate

interface InfoRepository {
    suspend fun getAvailableCinemas(): List<Cinema>
    suspend fun getAvailableFilms(): List<Film>
    suspend fun getTimetables(cinemaId: String, date: LocalDate): List<Timetable>
}