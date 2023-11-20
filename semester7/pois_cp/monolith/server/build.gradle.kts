plugins {
    alias(libs.plugins.kotlinJvm)
    alias(libs.plugins.ktor)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.shadow.jar)
    alias(libs.plugins.detekt)
    application
}

group = "ru.kheynov.hotel"
version = "1.0.0"
application {
    mainClass.set("ru.kheynov.hotel.ApplicationKt")
    applicationDefaultJvmArgs = listOf("-Dio.ktor.development=${extra["development"] ?: "false"}")
}

dependencies {
    implementation(projects.shared)
    implementation(libs.logback)
    implementation(libs.ktor.server.core)
    implementation(libs.ktor.server.netty)
    testImplementation(libs.ktor.server.tests)
    testImplementation(libs.kotlin.test.junit)

    implementation(libs.ktor.server.auth)
    implementation(libs.ktor.server.auth.jwt)
    implementation(libs.ktor.default.headers)
    implementation(libs.ktor.conditional.headers)
    implementation(libs.ktor.call.logging)
    implementation(libs.ktor.content.negotiation)
    implementation(libs.ktor.serialization)
    implementation(libs.ktor.websockets)
    implementation(libs.bcrypt)
    implementation(libs.ktorm.core)
    implementation(libs.ktorm.pg)
    implementation(libs.postgres)
}