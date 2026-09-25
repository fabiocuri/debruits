<script>
export default {
    name: 'Header',
    props: {
        title: String,
        action: String,
        color: String,
        bg: String
    },
    methods: {
        handleClick() {
            console.log("button clicked")
        }
    }
}
</script>

<template>

    <header>
        <h1 :style="{ color: color}">{{ title }}</h1>
        <button @click="handleClick()" :style="{ background: bg}">{{ action }}</button>
    </header>

</template>

<style>

button {

padding:10px 20px;
font-size:30px;
background: white;
cursor:pointer;
border: white;
}

h1 {

    color: black;
}

</style>