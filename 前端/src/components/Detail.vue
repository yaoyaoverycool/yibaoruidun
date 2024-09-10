<script setup lang="ts">
import { ref, reactive } from 'vue';
import axios from 'axios';
const API_BASE_URL = 'http://8.130.128.77:8848';


const data = ref([]);
const loading = ref(false);
const data1 = ref([]);
const file = ref(null);
const errorMessage = ref('');
const tableData = ref([])
const uploadFile = async () => {

  if (!file.value) {
    errorMessage.value = '没有选择文件';
    return;
  }
  loading.value = true
  const formData = new FormData();
  formData.append('file', file.value);

  try {
    const response = await fetch(`${API_BASE_URL}/upload_file`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json();
      errorMessage.value = errorData.error || '上传失败';
      return;
    }
    loading.value = false
    const responseData = await response.json();
    errorMessage.value = '';

    data1.value = responseData
    tableData.value = data1.value[0]

    tableData.value.push(...data1.value[1]);


  } catch (error) {
    errorMessage.value = '文件格式错误';
  }
};

const fileInputId = 'fileInput';
const fileName = ref('上传表格文件');

const handleFileUpload = (event) => {
  const fileInput = event.target;
  if (fileInput.files && fileInput.files.length > 0) {
    fileName.value = fileInput.files[0].name;
    file.value = fileInput.files[0];
  } else {
    fileName.value = '上传表格文件';
  }
};
const scrollContainer = ref(null);

import { ElTable } from 'element-plus'

const multipleTableRef = ref()
const selectedData = ref([])
const handleSelectionChange = (val) => {
  selectedData.value = val
}
let i = 2
const load = () => {
  tableData.value.push(...data1.value[i]);
  i++
}

import XLSX from 'xlsx';
const exportToExcel = (e) => {
  const fixedHeaders = ["有诈骗可能的投保人", "就诊次数", "月统筹金额", "月药品金额", "总金额", "可用账户报销金额"];
  if (Object.keys(e[0]).length !== fixedHeaders.length) {
    throw new Error('Invalid data. The number of object properties does not match the number of predefined headers.');
  }
  const dataWithHeaders = [fixedHeaders].concat(e.map(row => Object.values(row)));
  const columnWidths = [20, 15, 20, 20, 20, 25];
  const ws = XLSX.utils.aoa_to_sheet(dataWithHeaders);
  if (!ws['!cols']) {

    ws['!cols'] = [];

  }
  for (let i = 0; i < columnWidths.length; i++) {
    ws['!cols'][i] = { wch: columnWidths[i] };
  }
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Selected Data');
  const blob = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
  const url = window.URL.createObjectURL(new Blob([blob], { type: 'application/octet-stream' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'selected_data.xlsx';
  link.click();
  window.URL.revokeObjectURL(url);
};

const exportALLToExcel = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/upload_file`);
    if (response.status === 200) {
      data.value = response.data;
      exportToExcel(data.value)
    } else {
      console.error('Failed to fetch data:', response.statusText);
    }
  } catch (error) {
    console.error('Error fetching data:', error.message);
  }

}







import { onUnmounted, onMounted, watchEffect } from 'vue'
const windowWidth = ref(window.innerWidth);
const windowHeight = ref(window.innerHeight);
const handleResize = () => {
  windowWidth.value = window.innerWidth;
  windowHeight.value = window.innerHeight;
};
onMounted(() => {
  window.addEventListener('resize', handleResize);
  handleResize();
});

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
});

watchEffect(() => {
  if (windowWidth.value < 1408) {
    loading.value = false
  }
});

</script>

<template>
  <div class="main">
    <div class=" upfile_big_box">
      <div class="picture">
        方法一: 文件上传预测
      </div>
      <div class="loading_box" v-show="loading">加载中，请稍后 . . . . .</div>
      <div class="custom-input-container">
        <label :for="fileInputId" class="custom-file-label">{{ fileName }}</label>
        <input type="file" :id="fileInputId" class="custom-file-input" @change="handleFileUpload">
        <button class="custom-button" @click="uploadFile" >上传</button>
      </div>
      <div class="scroll_box" v-if="data1.length > 0">
        <div class="scroll_main">
          <div style="display: flex; align-items: center;justify-content: left;">
            <button @click="exportToExcel(selectedData)" class="exportToExcel">导出选中数据</button>
            <button @click="exportALLToExcel" class="exportALLToExcel">导出全部数据
            </button>
          </div>
          <div ref="scrollContainer" class="scroll_x">
            <el-table ref="multipleTableRef" :data="tableData" style="width: 100%"
              @selection-change="handleSelectionChange" height="500" border stripe v-el-table-infinite-scroll="load">
              <el-table-column type="selection" width="55" />
              <el-table-column property="Index" label="有诈骗可能的投保人" width="200" />
              <el-table-column property="Number_of_medical_visits_SUM" label="就诊次数" width="120" />
              <el-table-column property="Monthly_pooled_amount_MAX" label="月统筹金额" show-overflow-tooltip width="200" />
              <el-table-column property="Monthly_drug_amount_AVG" label="月药品金额" width="200" />
              <el-table-column property="ALL_SUM" label="总金额" width="200" />
              <el-table-column property="Available_account_reimbursement_amount_SUM" label="可用账户报销金额" width="200" />
            </el-table>
            
          </div>
          
        </div>
      </div>
      <div v-if="errorMessage !== ''" class="errorMessage_box">
        <p class="error-message">Error: {{ errorMessage }}</p>
      </div>
    </div>
  </div>


</template>


<style scoped>
.exportToExcel,
.exportALLToExcel {
  width: 100px;
  height: 40.8px;
  background-color: #BED0A6;
  color: white;
  border: none;
  margin: 10px 10px;
  border-radius: 5px;
  cursor: pointer;
}

.flex_box {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 1100px;
  height: 40px;
  margin: 5px auto;
}

.scroll_box_header {
  width: 1200px;
  height: 40px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.scroll_x {
  width: 1200px;
  height: 550px;

}

.loading_box {
  width: 260px;
  height: 48px;
  padding: 20px;
  background-color: #f0f0f0;
  position: absolute;
  top: 412px;
  left: 50%;
  translate: -50% 0;
  border-radius: 5px;
  display: flex;
  justify-self: center;
  align-items: center;

}
.scroll_box {
  width: 1200px;
  height:550px;
  z-index: 0;
}

.gap {
  width: 1200px;
  height: 5px;
  background-color: #77b396;
  margin: 30px auto;
}

.main {
  width: 1300px;

  margin: 0 auto;
  background-color: #f4f9f3;
  box-shadow: 0 4px 4px rgba(0, 0, 0, 0.4);
}


.upfile_big_box {
  background-color: #f4f9f3;

  width: 1300px;
  min-height: 595px;
  padding: 50px;
  background-image: url('@/store/detial.png');
  background-repeat: no-repeat;
  background-size: 100%;
}


.picture {
  margin: 30px auto;
  width: 400px;
  height: 200px;
  display: flex;
  align-items: end;
  justify-content: center;
  font-size: 30px;
  color: #1f9792;
  font-weight: 800;
  font-family: Black;
}


.custom-input-container {
  margin-left: auto;
  margin-right: auto;
  display: flex;
  align-items: center;
  margin-bottom: 30px;
  width: 300px;
  background-color: white;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}


.custom-file-label {
  background-color: #F0F0F0;
  padding: 10px 20px;
  border: none;
  border-bottom-left-radius: 5px;
  border-top-left-radius: 5px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  transition: background-color 0.2s ease-in-out;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  width: 200px;
  height: 40.8px;
}


.custom-file-input {
  display: none;
}


.custom-button {
  width: 72px;
  height: 40.8px;
  background-color: #BED0A6;
  color: white;
  border: none;

  border-radius: 0 5px 5px 0;
  cursor: pointer;
  transition: background-color 0.2s ease-in-out;


}


.custom-button:hover {
  background-color: #bcd0a1;
}

.predictions_table {
  margin-left: auto;
  margin-right: auto;
  width: 100%;
  border-collapse: collapse;
  margin-top: 30px;
}


.predictions_table th,
.predictions_table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}


.predictions_table th {
  background-color: #F0F0F0;
  font-weight: bold;
  color: #333;
}


.errorMessage_box {
  width: 200px;
  padding: 10px;
  background-color: #FFEAEA;
  border: 1px solid red;
  margin-top: 20px;
  text-align: center;
  margin-left: auto;
  margin-right: auto;
}


.error-message {

  color: red;
  font-weight: bold;
  background-color: #FFEAEA;
}
</style>
