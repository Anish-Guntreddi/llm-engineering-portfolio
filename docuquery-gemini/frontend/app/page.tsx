import Chat from "@/components/Chat";
import DocumentSidebar from "@/components/DocumentSidebar";

export default function HomePage() {
  return (
    <div className="mx-auto grid h-[calc(100vh-65px)] max-w-7xl grid-cols-1 gap-6 px-4 py-6 lg:grid-cols-[300px_1fr]">
      <div className="hidden min-h-0 lg:block">
        <DocumentSidebar />
      </div>

      {/* Mobile: collapsible-ish stacked sidebar above chat */}
      <div className="lg:hidden">
        <DocumentSidebar />
      </div>

      <div className="min-h-0">
        <Chat />
      </div>
    </div>
  );
}
