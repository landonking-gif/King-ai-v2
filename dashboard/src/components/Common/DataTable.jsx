import { useState, useMemo } from 'react';
import './Common.css';

export function DataTable({ columns, data, onRowClick, pageSize = 10 }) {
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(0);
  
  const sortedData = useMemo(() => {
    if (!sortColumn) return data;
    
    return data.slice().sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];
      
      if (typeof aVal === 'number') {
        return sortDir === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      const cmp = String(aVal).localeCompare(String(bVal));
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, sortColumn, sortDir]);
  
  const pagedData = useMemo(() => {
    const start = page * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, page, pageSize]);
  
  const totalPages = Math.ceil(data.length / pageSize);
  
  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDir('asc');
    }
  };
  
  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => col.sortable !== false && handleSort(col.key)}
                className={col.sortable !== false ? 'sortable' : ''}
              >
                {col.label}
                {sortColumn === col.key && (
                  <span className="sort-icon">
                    {sortDir === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        
        <tbody>
          {pagedData.map((row, i) => (
            <tr
              key={row.id || i}
              onClick={() => onRowClick?.(row)}
              className={onRowClick ? 'clickable' : ''}
            >
              {columns.map((col) => (
                <td key={col.key}>
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
          
          {pagedData.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="empty-row">
                No data available
              </td>
            </tr>
          )}
        </tbody>
      </table>
      
      {totalPages > 1 && (
        <div className="table-pagination">
          <button
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </button>
          
          <span className="page-info">
            Page {page + 1} of {totalPages}
          </span>
          
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
