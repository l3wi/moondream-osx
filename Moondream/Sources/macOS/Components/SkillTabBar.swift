import SwiftUI
import MoondreamKit

/// Tab bar for selecting between different skills
struct SkillTabBar: View {
    @Binding var selection: Skill

    var body: some View {
        HStack(spacing: 4) {
            ForEach(Skill.allCases) { skill in
                SkillTab(skill: skill, isSelected: selection == skill)
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selection = skill
                        }
                    }
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
}

/// Individual skill tab button
struct SkillTab: View {
    let skill: Skill
    let isSelected: Bool
    @State private var isHovered = false

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: skill.icon)
                .font(.system(size: 16))

            Text(skill.displayName)
                .font(.caption)
                .fontWeight(isSelected ? .semibold : .regular)
        }
        .foregroundStyle(isSelected ? .primary : .secondary)
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.accentColor.opacity(0.15) : (isHovered ? Color.secondary.opacity(0.1) : .clear))
        )
        .contentShape(Rectangle())
        .onHover { hovering in
            withAnimation(.easeInOut(duration: 0.15)) {
                isHovered = hovering
            }
        }
    }
}

#Preview {
    SkillTabBar(selection: .constant(.caption))
        .padding()
}
