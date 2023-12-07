package ru.kheynov.hotel.data.repositories

import org.ktorm.database.Database
import org.ktorm.dsl.eq
import org.ktorm.dsl.from
import org.ktorm.dsl.innerJoin
import org.ktorm.dsl.map
import org.ktorm.dsl.select
import org.ktorm.dsl.where
import ru.kheynov.hotel.data.entities.Hotels
import ru.kheynov.hotel.data.entities.Reservations
import ru.kheynov.hotel.data.entities.Rooms
import ru.kheynov.hotel.domain.entities.Hotel
import ru.kheynov.hotel.domain.entities.Room
import ru.kheynov.hotel.domain.entities.RoomReservation
import ru.kheynov.hotel.domain.repository.ReservationsRepository

class PostgresReservationsRepository(
    private val database: Database
) : ReservationsRepository {

    override suspend fun getAvailableRooms(hotel: Hotel): List<RoomReservation> {
        return database
            .from(Rooms)
            .innerJoin(Reservations, on = Rooms.id eq Reservations.room)
            .select()
            .where { Rooms.hotel eq hotel.id }
            .map { row ->
                RoomReservation(
                    id = row[Reservations.id]!!,
                    room = Room(
                        id = row[Rooms.id]!!,
                        type = row[Rooms.type]!!,
                        price = row[Rooms.price]!!,
                        hotel = hotel
                    ),
                    from = row[Reservations.arrivalDate]!!,
                    to = row[Reservations.departureDate]!!
                )
            }
    }

    override suspend fun getHotels(): List<Hotel> {
        return database
            .from(Hotels)
            .select()
            .map { row ->
                Hotel(
                    id = row[Hotels.id]!!,
                    name = row[Hotels.name]!!,
                    city = row[Hotels.city]!!,
                    address = row[Hotels.address]!!,
                    stars = row[Hotels.rating]!!
                )
            }
    }

    override suspend fun reserveRoom(room: RoomReservation): Boolean {
        TODO("Not yet implemented")
    }

    override suspend fun cancelReservation(reservationId: String): Boolean {
        TODO("Not yet implemented")
    }
}