plugins {
    java
    kotlin("jvm") version "1.4.10"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("com.github.doyaaaaaken:kotlin-csv-jvm:0.12.0")
    testCompile("junit", "junit", "4.12")
}
