import NavItem from './NavItem'

export default function NavTree({ items, level = 0, onNavigate }) {
  if (!items || items.length === 0) return null

  return (
    <ul className={`nav-tree level-${level}`}>
      {items.map(item => (
        <NavItem 
          key={item.id} 
          item={item} 
          level={level}
          onNavigate={onNavigate}
        />
      ))}
    </ul>
  )
}

