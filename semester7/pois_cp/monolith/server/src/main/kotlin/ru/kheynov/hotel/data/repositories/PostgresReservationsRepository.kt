package ru.kheynov.hotel.data.repositories

import org.ktorm.database.Database
import org.ktorm.dsl.delete
import org.ktorm.dsl.eq
import org.ktorm.dsl.from
import org.ktorm.dsl.innerJoin
import org.ktorm.dsl.insert
import org.ktorm.dsl.map
import org.ktorm.dsl.select
import org.ktorm.dsl.where
import ru.kheynov.hotel.data.entities.Hotels
import ru.kheynov.hotel.data.entities.Reservations
import ru.kheynov.hotel.data.entities.Rooms
import ru.kheynov.hotel.data.entities.Users
import ru.kheynov.hotel.shared.domain.entities.Hotel
import ru.kheynov.hotel.shared.domain.entities.Room
import ru.kheynov.hotel.shared.domain.entities.RoomInfo
import ru.kheynov.hotel.shared.domain.entities.RoomReservation
import ru.kheynov.hotel.shared.domain.entities.RoomReservationInfo
import ru.kheynov.hotel.shared.domain.entities.User
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository
import java.time.LocalDate

class PostgresReservationsRepository(
    private val database: Database
) : ReservationsRepository {
    override suspend fun getRooms(hotel: Hotel): List<RoomInfo> {
        return database
            .from(Rooms)
            .select()
            .where { Rooms.hotel eq hotel.id }
            .map { row ->
                RoomInfo(
                    id = row[Rooms.id]!!,
                    type = row[Rooms.type]!!,
                    price = row[Rooms.price]!!,
                )
            }
    }

    override suspend fun getOccupiedRooms(hotel: Hotel): List<RoomReservationInfo> {
        return database
            .from(Rooms)
            .innerJoin(Reservations, on = Rooms.id eq Reservations.room)
            .select()
            .where { Rooms.hotel eq hotel.id }
            .map { row ->
                RoomReservationInfo(
                    id = row[Reservations.id]!!,
                    room = Room(
                        id = row[Rooms.id]!!,
                        type = row[Rooms.type]!!,
                        price = row[Rooms.price]!!,
                        hotel = hotel
                    ),
                    from = row[Reservations.arrivalDate]!!,
                    to = row[Reservations.departureDate]!!,
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

    override suspend fun getRoomByID(id: String): Room? {
        return database
            .from(Rooms)
            .select()
            .where { Rooms.id eq id }
            .map { row ->
                Room(
                    id = row[Rooms.id]!!,
                    type = row[Rooms.type]!!,
                    price = row[Rooms.price]!!,
                    hotel = getHotels().find { it.id == row[Rooms.hotel]!! }!!
                )
            }
            .firstOrNull()
    }

    override suspend fun getRoomOccupancy(roomId: String): List<ClosedRange<LocalDate>> {
        return database
            .from(Reservations)
            .select()
            .where { Reservations.room eq roomId }
            .map { row ->
                row[Reservations.arrivalDate]!!..row[Reservations.departureDate]!!
            }
    }

    override suspend fun reserveRoom(room: RoomReservation): Boolean {
        return database.insert(Reservations) {
            set(it.id, room.id)
            set(it.room, room.room.id)
            set(it.arrivalDate, room.from)
            set(it.departureDate, room.to)
            set(it.guest, room.user.id)
        } > 0
    }

    override suspend fun getReservationByID(reservationId: String): RoomReservation? {
        return database
            .from(Reservations)
            .innerJoin(Users, on = Reservations.guest eq Users.userId)
            .select()
            .where { Reservations.id eq reservationId }
            .map { row ->
                RoomReservation(
                    id = row[Reservations.id]!!,
                    room = getRoomByID(row[Reservations.room]!!)!!,
                    from = row[Reservations.arrivalDate]!!,
                    to = row[Reservations.departureDate]!!,
                    user = User(
                        id = row[Users.userId]!!,
                        name = row[Users.name]!!,
                        email = row[Users.email]!!
                    )
                )
            }
            .firstOrNull()
    }

    override suspend fun getUsersReservations(user: User): List<RoomReservationInfo> {
        return database
            .from(Reservations)
            .innerJoin(Rooms, on = Reservations.room eq Rooms.id)
            .select()
            .where { Reservations.guest eq user.id }
            .map { row ->
                RoomReservationInfo(
                    id = row[Reservations.id]!!,
                    room = Room(
                        id = row[Rooms.id]!!,
                        type = row[Rooms.type]!!,
                        price = row[Rooms.price]!!,
                        hotel = getHotels().find { it.id == row[Rooms.hotel]!! }!!
                    ),
                    user = user,
                    from = row[Reservations.arrivalDate]!!,
                    to = row[Reservations.departureDate]!!,
                )
            }
    }


    override suspend fun cancelReservation(reservationId: String): Boolean {
        return database.delete(Reservations) {
            it.id eq reservationId
        } > 0
    }
}