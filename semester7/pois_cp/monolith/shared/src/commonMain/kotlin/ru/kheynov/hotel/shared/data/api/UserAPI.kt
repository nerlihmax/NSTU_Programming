package ru.kheynov.hotel.shared.data.api

import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.PATCH
import retrofit2.http.POST
import retrofit2.http.Query
import ru.kheynov.hotel.shared.data.models.users.UpdateUserRequest
import ru.kheynov.hotel.shared.data.models.users.auth.LoginViaEmailRequest
import ru.kheynov.hotel.shared.data.models.users.auth.RefreshTokenRequest
import ru.kheynov.hotel.shared.data.models.users.auth.SignUpViaEmailRequest
import ru.kheynov.hotel.shared.domain.entities.User
import ru.kheynov.hotel.shared.domain.entities.UserEmployment
import ru.kheynov.hotel.shared.jwt.TokenPair

interface UserAPI {
    @GET("user")
    suspend fun getUserInfo(
        @Query("id") id: String? = null,
    ): User

    @PATCH("user")
    suspend fun updateUserInfo(
        @Body user: UpdateUserRequest,
    )

    @GET("user/employee")
    suspend fun getUserEmployment(): UserEmployment


    @POST("auth/email/register")
    suspend fun createUser(
        @Body body: SignUpViaEmailRequest,
        @Header("client-id") clientId: String
    ): TokenPair

    @POST("auth/email/login")
    suspend fun loginUser(
        @Body loginUser: LoginViaEmailRequest,
        @Header("client-id") clientId: String
    ): TokenPair

    @POST("auth/refresh")
    suspend fun refreshToken(
        @Body refreshToken: RefreshTokenRequest,
        @Header("client-id") clientId: String
    ): TokenPair
}