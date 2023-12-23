package ru.kheynov.hotel.shared.data.repository

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import retrofit2.HttpException
import ru.kheynov.hotel.shared.data.api.UserAPI
import ru.kheynov.hotel.shared.data.models.users.UpdateUserRequest
import ru.kheynov.hotel.shared.data.models.users.auth.LoginViaEmailRequest
import ru.kheynov.hotel.shared.data.models.users.auth.RefreshTokenRequest
import ru.kheynov.hotel.shared.data.models.users.auth.SignUpViaEmailRequest
import ru.kheynov.hotel.shared.domain.entities.User
import ru.kheynov.hotel.shared.jwt.TokenPair
import ru.kheynov.hotel.shared.utils.BadRequestException
import ru.kheynov.hotel.shared.utils.ForbiddenException
import ru.kheynov.hotel.shared.utils.NetworkException
import ru.kheynov.hotel.shared.utils.ServerSideException
import ru.kheynov.hotel.shared.utils.UnauthorizedException
import java.net.HttpURLConnection.HTTP_BAD_GATEWAY
import java.net.HttpURLConnection.HTTP_BAD_REQUEST
import java.net.HttpURLConnection.HTTP_FORBIDDEN
import java.net.HttpURLConnection.HTTP_INTERNAL_ERROR
import java.net.HttpURLConnection.HTTP_UNAUTHORIZED


class ClientUsersRepository(
    private val usersAPI: UserAPI,
) {
    suspend fun registerUser(
        data: SignUpViaEmailRequest,
        clientId: String,
    ): Result<TokenPair> =
        withContext(Dispatchers.IO) {
            try {
                Result.success(usersAPI.createUser(data, clientId))
            } catch (e: Exception) {
                handleException(e)
            }
        }

    suspend fun getUserInfo(): Result<User?> {
        return try {
            val result = usersAPI.getUserInfo().let { User(it.id, it.name, it.email) }
            Result.success(result)
        } catch (e: Exception) {
            handleException(e)
        }
    }

    suspend fun getUserInfoByID(id: String): Result<User?> {
        return try {
            val result = usersAPI.getUserInfo(id).let { User(it.id, it.name, it.email) }
            Result.success(result)
        } catch (e: Exception) {
            handleException(e)
        }
    }

    suspend fun updateUser(update: UpdateUserRequest): Result<Boolean> {
        return try {
            usersAPI.updateUserInfo(update)
            Result.success(true)
        } catch (e: Exception) {
            handleException(e)
        }
    }

    suspend fun loginUser(
        data: LoginViaEmailRequest,
        clientId: String,
    ): Result<TokenPair> =
        withContext(Dispatchers.IO) {
            try {
                Result.success(usersAPI.loginUser(data, clientId))
            } catch (e: Exception) {
                handleException(e)
            }
        }

    suspend fun refreshToken(
        oldRefreshToken: String,
        clientId: String,
    ): Result<TokenPair> =
        withContext(Dispatchers.IO) {
            try {
                val result =
                    usersAPI.refreshToken(RefreshTokenRequest(oldRefreshToken), clientId)
                Result.success(result)
            } catch (e: Exception) {
                handleException(e)
            }
        }

    private fun handleException(e: Exception): Result<Nothing> {
        return if (e is HttpException) {
            Result.failure(
                e.response()?.errorBody()?.string().toString()
                    .let {
                        when (e.code()) {
                            HTTP_BAD_REQUEST -> when {
                                else -> BadRequestException()
                            }

                            HTTP_INTERNAL_ERROR, HTTP_BAD_GATEWAY -> ServerSideException()
                            HTTP_UNAUTHORIZED -> UnauthorizedException()
                            HTTP_FORBIDDEN -> ForbiddenException()
                            else -> e
                        }
                    }
            )
        } else if (e.message?.let { it.contains("hostname") || it.contains("timeout") } == true) {
            Result.failure(NetworkException())
        } else {
            Result.failure(e)
        }
    }
}
