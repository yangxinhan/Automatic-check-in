class AttendanceSystem {
    constructor() {
        this.records = [];
        this.autoRefreshInterval = 30000; // 30秒更新一次
        this.initializeSystem();
    }

    initializeSystem() {
        this.loadRecords();
        this.startAutoRefresh();
        this.initializeEventListeners();
        this.initializeList();
    }

    async loadRecords() {
        try {
            const response = await fetch('attendance.csv');
            if (!response.ok) throw new Error('無法載入檔案');
            const data = await response.text();
            this.records = this.parseCSV(data);
            this.displayRecords(this.records);
        } catch (error) {
            console.error('載入記錄失敗:', error);
            this.showError('載入記錄失敗，請重新整理頁面');
        }
    }

    parseCSV(csv) {
        const lines = csv.split('\n');
        return lines.slice(1)
            .filter(line => line.trim())
            .map(line => {
                const [name, checkIn, checkOut, status] = line.split(',');
                return { 
                    name, 
                    checkIn: new Date(checkIn), 
                    checkOut: checkOut ? new Date(checkOut) : null, 
                    status 
                };
            })
            .sort((a, b) => b.checkIn - a.checkIn); // 預設按時間倒序
    }

    displayRecords(records) {
        const tbody = document.getElementById('recordsTable');
        tbody.innerHTML = '';
        
        records.forEach(record => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${record.name}</td>
                <td>${this.formatDate(record.checkIn)}</td>
                <td>${record.checkOut ? this.formatDate(record.checkOut) : '-'}</td>
                <td>
                    <span class="badge bg-${record.status === '簽到' ? 'success' : 'info'}">
                        ${record.status}
                    </span>
                </td>
            `;
            tbody.appendChild(row);
        });

        this.updateStats(records);
    }

    formatDate(date) {
        return new Intl.DateTimeFormat('zh-TW', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }).format(date);
    }

    searchRecords() {
        const nameQuery = document.getElementById('nameSearch').value.toLowerCase();
        const dateQuery = document.getElementById('dateSearch').value;
        const statusQuery = document.getElementById('statusFilter').value;

        const filtered = this.records.filter(record => {
            const nameMatch = !nameQuery || record.name.toLowerCase().includes(nameQuery);
            const dateMatch = !dateQuery || this.formatDate(record.checkIn).includes(dateQuery);
            const statusMatch = !statusQuery || record.status === statusQuery;
            return nameMatch && dateMatch && statusMatch;
        });

        this.displayRecords(filtered);
    }

    updateStats(records) {
        const today = new Date().toDateString();
        const todayRecords = records.filter(r => r.checkIn.toDateString() === today);
        
        const stats = {
            total: todayRecords.length,
            present: todayRecords.filter(r => r.status === '簽到' && !r.checkOut).length
        };

        document.getElementById('totalCount').textContent = stats.total;
        document.getElementById('presentCount').textContent = stats.present;
    }

    startAutoRefresh() {
        setInterval(() => this.loadRecords(), this.autoRefreshInterval);
    }

    initializeEventListeners() {
        // 搜尋功能
        document.getElementById('nameSearch').addEventListener('input', () => this.searchRecords());
        document.getElementById('dateSearch').addEventListener('change', () => this.searchRecords());
        document.getElementById('statusFilter').addEventListener('change', () => this.searchRecords());

        // 排序功能
        document.querySelectorAll('th[data-sort]').forEach(header => {
            header.addEventListener('click', () => this.sortRecords(header.dataset.sort));
        });
    }

    sortRecords(field) {
        this.records.sort((a, b) => {
            if (field === 'name') return a.name.localeCompare(b.name);
            if (field === 'time') return b.checkIn - a.checkIn;
            return 0;
        });
        this.displayRecords(this.records);
    }

    showError(message) {
        // 可以改為使用 Toast 或其他 UI 元件
        alert(message);
    }

    initializeList() {
        const list1 = document.getElementById('studentList1');
        const list2 = document.getElementById('studentList2');
        
        // 生成1-20號學生列表
        for (let i = 1; i <= 20; i++) {
            list1.appendChild(this.createStudentListItem(i));
        }
        
        // 生成21-40號學生列表
        for (let i = 21; i <= 40; i++) {
            list2.appendChild(this.createStudentListItem(i));
        }
    }

    createStudentListItem(number) {
        const item = document.createElement('div');
        item.className = 'list-group-item d-flex justify-content-between align-items-center';
        item.id = `student-list-${number}`;
        
        item.innerHTML = `
            <div>
                <span class="badge bg-secondary rounded-pill me-2">${number}</span>
                <span>${number}號</span>
            </div>
            <span class="badge bg-danger">未到</span>
        `;
        
        return item;
    }

    updateStudentStatus(studentNumber, isPresent) {
        // 更新原有的座位圖
        const seat = document.getElementById(`student-${studentNumber}`);
        if (seat) {
            seat.className = isPresent ? 'seat present' : 'seat absent';
        }
        
        // 更新列表狀態
        const listItem = document.getElementById(`student-list-${studentNumber}`);
        if (listItem) {
            const statusBadge = listItem.querySelector('.badge:last-child');
            statusBadge.className = isPresent ? 
                'badge bg-success' : 
                'badge bg-danger';
            statusBadge.textContent = isPresent ? '已到' : '未到';
        }
    }
}

// 初始化系統
document.addEventListener('DOMContentLoaded', () => {
    window.attendanceSystem = new AttendanceSystem();
});