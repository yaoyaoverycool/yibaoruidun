<script setup>

</script>
<template>
  <div class="home">
      <div class="img_box">
        <div class="btn_box">
          <RouterLink to="/upload_file" class="btn">文件上传预测</RouterLink>
          <RouterLink to="/upload_text" class="btn" >文本输入预测</RouterLink>
        </div>
      </div>
  </div>
</template>
<style scoped>

.btn{
  display: block;
  text-align: center;
  width: 160px;
  height: 46.4px;
  line-height: 46.4px;
  background-color: #1f9792;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 20px;
  font-weight: 600;
  font-family: black;
}
.btn_box{
  display: flex;
  justify-self: center;
  align-items: center;
  justify-content: space-between;
  align-items: center;
  height: 60px;
  width: 400px;
  margin: 430px auto;
}
.home{
  width: 1300px;
  margin: 0 auto;
}
.img_box{
  display: flex;
  justify-items: center;
  width: 1300px;
  height: 615px;
  background-image: url(@/store/home1.png);
  background-size: 100% 100%;
  background-repeat: no-repeat;
}
</style>