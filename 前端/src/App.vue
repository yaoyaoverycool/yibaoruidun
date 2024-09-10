<script setup>
import { onMounted } from 'vue';


onMounted(() => {
  const keys = [];

  for (let i = 0; i < localStorage.length; i++) {
    keys.push(localStorage.key(i));
  }

  keys.forEach((key) => {
    localStorage.removeItem(key);
  });

});

</script>

<template>
  <div class="app_box">
    <div class="header_unfixed">
      <div class="header">
      <div class="header_nav">
        <div class="header_nav_l">
          <img src="./store/logo1.png" alt="" style="height: 72px;">
        </div>
        <div class="header_nav_r">
          <RouterLink to="/" active-class="active" style="color: black;">主页</RouterLink>
          <RouterLink to="/upload_file" active-class="active" style="color: black;">文件预测</RouterLink>
          <RouterLink to="/upload_text" active-class="active" style="color: black;">文本预测</RouterLink>
        </div>
      </div>
    </div>
    </div>
    
    <div class="main_content">
      <router-view></router-view>
    </div>
  </div>
</template>


<style scoped>
.header_unfixed{
  width: 100%;
  height: 84px;
  box-shadow: 0px -4px 5px 2px black;
  background-color: white;
}
.header{
  display: flex;
  position: fixed;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 84px;
  background-color: white;
  box-shadow: 0px -4px 5px 2px black;
  z-index: 999;
}
.header_nav{
  display: flex;
  background-color: white;

  justify-content: space-between;
  align-items: center;
  width: 1400px;
  height: 84px;
}
.header_nav_l{
  display: flex;
  align-items: center;
}
.header_nav_r{
  display: flex;
  align-items: center;
  margin-right: 30px;
  font-size: 26px;
  font-weight: 500;
  width: 450px;
  height: 134px;
}
.header_nav_r a{
  line-height: 84px;
  height: 84px;
  margin: 25px;
  background-color: white
  
}
.active{
  border-bottom: 3px solid green;
}
.main_content{

  width: 1300px;
  margin: 2px auto;
}
</style>

